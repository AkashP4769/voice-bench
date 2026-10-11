// Realtime transcription page.
//
// Flow: open a WebSocket, send a start frame, then stream raw PCM as binary
// frames while the server relays each model's partial and final events back.
//
// Two deliberate choices:
//   * The AudioContext is created at 16 kHz so the browser resamples the
//     microphone for us. That removes resampling from both the client and the
//     server. If a browser refuses the rate, the real rate is sent instead and
//     the server-side path still refuses it loudly rather than guessing.
//   * Frames are small (250 ms) so Phonon's own partials can appear at their
//     natural 0.5 s cadence. Whistle just ignores frames until it has enough
//     audio for one pass, which is why 2 s is the default cadence.

(function () {
  "use strict";

  const PROCESSOR_NAME = "voicebench-recorder";
  const TARGET_SAMPLE_RATE = 16000;
  const FRAME_SAMPLES = 4000; // 250 ms
  const MAX_SECONDS = 600;
  const SCRIPT_VERSION = (() => {
    const tag = document.currentScript;
    return tag && tag.dataset ? tag.dataset.version || "" : "";
  })();
  const WORKLET_URL =
    "/static/recorder-worklet.js" + (SCRIPT_VERSION ? "?v=" + SCRIPT_VERSION : "");

  const dash = "\u2014";

  function seconds(value) {
    return Number.isFinite(value) ? value.toFixed(1) + " s" : dash;
  }

  function millis(value) {
    return Number.isFinite(value) ? Math.round(value) + " ms" : dash;
  }

  function controlState() {
    return {
      providers: Array.from(
        document.querySelectorAll('input[name="rt-provider"]:checked')
      ).map((box) => box.value),
      language: (document.getElementById("rt-language") || {}).value || null,
      cadence: parseFloat((document.getElementById("rt-cadence") || {}).value) || 2,
      keywords: (document.getElementById("rt-keywords") || {}).value || "",
      wordTimestamps: !!(document.getElementById("rt-word-timestamps") || {}).checked,
    };
  }

  function createController() {
    const recordButton = document.getElementById("rt-record");
    const recordLabel = document.getElementById("rt-record-label");
    const stopButton = document.getElementById("rt-stop");
    const clearButton = document.getElementById("rt-clear");
    const statusLine = document.getElementById("rt-status");
    const hint = document.getElementById("rt-hint");
    const panelsCard = document.getElementById("rt-panels-card");
    const panelsHost = document.getElementById("rt-panels");
    const finalCard = document.getElementById("rt-final-card");
    const finalHost = document.getElementById("rt-final");
    const template = document.getElementById("rt-panel-template");
    const saveButton = document.getElementById("rt-save");
    const copyButton = document.getElementById("rt-copy");
    const saveStatus = document.getElementById("rt-save-status");

    if (!recordButton) {
      return null;
    }
    const defaultHint = hint ? hint.textContent : "";

    const state = {
      phase: "idle", // idle | starting | live | stopping | done
      socket: null,
      context: null,
      stream: null,
      node: null,
      source: null,
      samplesSent: 0,
      startedAt: 0,
      ticking: null,
      panels: new Map(), // provider -> {elements, text}
      providers: [],
      options: null,
      results: null,
      implementations: {},
      pendingStop: null,
    };

    function setStatus(message, tone) {
      statusLine.textContent = message;
      statusLine.className = "source-status" + (tone ? " " + tone : "");
    }

    function refreshButtons() {
      const live = state.phase === "live" || state.phase === "starting";
      recordButton.disabled = live || state.providers.length === 0;
      stopButton.disabled = !live;
      clearButton.disabled = state.phase === "stopping";
      if (recordLabel) {
        recordLabel.textContent = live ? "Listening…" : "Start listening";
      }
      recordButton.classList.toggle("recording", live);
    }

    // -- panels ----------------------------------------------------------

    function buildPanels(providers, implementations) {
      panelsHost.innerHTML = "";
      state.panels.clear();
      providers.forEach((name) => {
        const fragment = template.content.cloneNode(true);
        const panel = fragment.querySelector(".panel");
        panel.dataset.provider = name;
        panel.querySelector(".panel-name").textContent = name;
        const impl = implementations && implementations[name];
        if (impl) {
          panel.querySelector(".panel-impl").textContent = impl;
        }
        panelsHost.appendChild(fragment);
        state.panels.set(name, {
          root: panelsHost.querySelector('[data-provider="' + CSS.escape(name) + '"]'),
          text: "",
        });
        setPanelState(name, "starting", null);
      });
      panelsCard.hidden = providers.length === 0;
    }

    function panelFor(name) {
      let panel = state.panels.get(name);
      if (!panel) {
        buildPanels(state.providers.concat([name]), state.implementations);
        panel = state.panels.get(name);
      }
      return panel;
    }

    function setPanelState(name, stateName, note) {
      const panel = panelFor(name);
      if (!panel || !panel.root) return;
      panel.root.dataset.state = stateName;
      panel.root.querySelector(".panel-state").textContent = stateName;
      const noteElement = panel.root.querySelector(".panel-note");
      if (note) {
        noteElement.hidden = false;
        noteElement.textContent = note;
      } else {
        noteElement.hidden = true;
      }
    }

    function setPanelText(name, text, pending) {
      const panel = panelFor(name);
      if (!panel || !panel.root) return;
      panel.text = text || "";
      panel.root.querySelector(".committed").textContent = text || "";
      const pendingElement = panel.root.querySelector(".pending");
      pendingElement.textContent = pending ? " " + pending : "";
    }

    function setPanelFacts(name, event) {
      const panel = panelFor(name);
      if (!panel || !panel.root) return;
      if (Number.isFinite(event.receivedS)) {
        panel.root.querySelector(".f-audio").textContent = seconds(event.receivedS);
      }
      if (Number.isFinite(event.passMs)) {
        const cadenceMs = (state.options ? state.options.cadence : 2) * 1000;
        panel.root.querySelector(".f-pass").textContent = millis(event.passMs);
        panel.root.querySelector(".f-cadence").textContent = seconds(cadenceMs / 1000);
        // "Keeping up" is the honest headline: a pass slower than the audio it
        // covers means this model is drifting behind realtime.
        const lag = panel.root.querySelector(".f-lag");
        const ratio = event.passMs / cadenceMs;
        lag.textContent = ratio <= 1 ? "yes (" + ratio.toFixed(2) + "\u00d7)" : "behind (" + ratio.toFixed(2) + "\u00d7)";
        lag.className = "f-lag " + (ratio <= 1 ? "ok" : "warn");
      }
      if (event.segment) {
        panel.root.querySelector(".f-segments").textContent = String(event.segment);
      }
    }

    // -- socket ----------------------------------------------------------

    function socketUrl() {
      const scheme = location.protocol === "https:" ? "wss:" : "ws:";
      return scheme + "//" + location.host + "/ws/realtime";
    }

    function handleMessage(raw) {
      let message;
      try {
        message = JSON.parse(raw);
      } catch (error) {
        return;
      }
      const name = message.provider;

      if (message.type === "ready") {
        state.providers = message.providers || [];
        state.implementations = message.implementations || {};
        if (message.failed) {
          Object.keys(message.failed).forEach((failed) => {
            const panel = panelFor(failed);
            if (panel && panel.root) {
              setPanelState(failed, "error", message.failed[failed]);
            }
          });
        }
        if (!state.providers.length) {
          setStatus("No model could start a live session.", "bad");
          return;
        }
        setStatus("Listening. Audio is being transcribed as you speak.", "live");
        return;
      }

      if (message.type === "status") {
        setPanelState(name, message.state || "streaming", message.detail || null);
        return;
      }
      if (message.type === "partial") {
        setPanelText(name, message.text, message.pending);
        setPanelFacts(name, message);
        return;
      }
      if (message.type === "final") {
        const panel = panelFor(name);
        if (panel) {
          setPanelFacts(name, message);
          // A final settles a phrase; the partial that follows carries the full
          // transcript again, so nothing needs concatenating here.
        }
        return;
      }
      if (message.type === "error") {
        setPanelState(name, "error", message.message);
        return;
      }
      if (message.type === "ended") {
        handleEnded(message);
      }
    }

    function handleEnded(message) {
      state.results = message;
      const transcripts = message.transcripts || {};
      const errors = message.errors || {};
      const metrics = message.metrics || {};

      finalHost.innerHTML = "";
      Object.keys(transcripts).forEach((name) => {
        const article = document.createElement("article");
        article.className = "transcript";
        const heading = document.createElement("h3");
        heading.textContent = name;
        article.appendChild(heading);
        if (errors[name]) {
          const failure = document.createElement("p");
          failure.className = "alert-inline";
          failure.textContent = errors[name];
          article.appendChild(failure);
        } else {
          const body = document.createElement("p");
          body.className = "text";
          body.textContent = transcripts[name] || "(no speech recognised)";
          article.appendChild(body);
        }
        const facts = document.createElement("p");
        facts.className = "muted";
        const own = metrics[name] || {};
        const bits = [];
        if (Number.isFinite(own.audio_s)) bits.push(seconds(own.audio_s) + " audio");
        if (Number.isFinite(own.mean_pass_ms)) bits.push("mean pass " + millis(own.mean_pass_ms));
        if (Number.isFinite(own.realtime_factor)) {
          bits.push(
            "decode " + own.realtime_factor.toFixed(2) + "\u00d7 realtime" +
              (own.realtime_factor > 1 ? " (fell behind)" : "")
          );
        }
        if (Number.isFinite(own.segments) && own.segments) bits.push(own.segments + " segments");
        facts.textContent = bits.join(" \u00b7 ");
        article.appendChild(facts);
        finalHost.appendChild(article);
      });

      finalCard.hidden = false;
      finalCard.scrollIntoView({ block: "nearest" });
      saveButton.disabled = false;
      saveStatus.textContent = "";
      state.phase = "done";
      clearTimers();
      teardown();
      setStatus("Stopped. Transcripts are below.", "ok");
      refreshButtons();
      // A pending automatic stop (client-initiated) resolves here.
      if (state.pendingStop) {
        const resolve = state.pendingStop;
        state.pendingStop = null;
        resolve(message);
      }
    }

    // -- capture ---------------------------------------------------------

    function clearTimers() {
      if (state.ticking) {
        clearInterval(state.ticking);
        state.ticking = null;
      }
    }

    function teardown() {
      if (state.node) {
        try {
          state.node.port.postMessage("close");
          state.node.disconnect();
        } catch (error) {
          /* already gone */
        }
        state.node = null;
      }
      if (state.source) {
        try {
          state.source.disconnect();
        } catch (error) {
          /* already gone */
        }
        state.source = null;
      }
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
      }
      if (state.context) {
        const context = state.context;
        state.context = null;
        context.close().catch(() => {});
      }
    }

    function describeError(error) {
      const name = error && error.name ? error.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        return "Microphone access was blocked. Allow it for this page, then try again.";
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        return "No microphone was found.";
      }
      if (name === "NotReadableError") {
        return "The microphone is in use by another application.";
      }
      return "Microphone error: " + (error && error.message ? error.message : String(error));
    }

    async function startWithStream(stream) {
      const control = controlState();
      if (!control.providers.length) {
        throw new Error("select at least one model");
      }
      state.options = control;
      state.providers = control.providers;
      state.samplesSent = 0;
      state.results = null;
      finalCard.hidden = true;
      saveButton.disabled = true;

      buildPanels(control.providers, {});
      setStatus("Opening the session…", null);

      const socket = new WebSocket(socketUrl());
      socket.binaryType = "arraybuffer";
      state.socket = socket;

      const opened = new Promise((resolve, reject) => {
        socket.onopen = () => resolve();
        socket.onerror = () => reject(new Error("could not open the realtime socket"));
      });
      socket.onmessage = (event) => handleMessage(event.data);
      socket.onclose = () => {
        if (state.phase === "live" || state.phase === "starting") {
          setStatus("The connection closed before the session ended.", "bad");
          state.phase = "idle";
          clearTimers();
          teardown();
          refreshButtons();
        }
      };

      await opened;
      socket.send(
        JSON.stringify({
          type: "start",
          providers: control.providers,
          language: control.language === "auto" ? null : control.language,
          keywords: control.keywords,
          wordTimestamps: control.wordTimestamps,
          cadenceSeconds: control.cadence,
          sampleRate: TARGET_SAMPLE_RATE,
        })
      );

      const context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
      if (context.state === "suspended") {
        await context.resume();
      }
      if (context.sampleRate !== TARGET_SAMPLE_RATE) {
        // Do not silently send audio at the wrong rate: the server would refuse
        // it, and the refusal is more useful than a wrong transcript.
        throw new Error(
          "this browser gave a " + context.sampleRate + " Hz audio context, but " +
            TARGET_SAMPLE_RATE + " Hz is required for realtime"
        );
      }

      await context.audioWorklet.addModule(WORKLET_URL);
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, PROCESSOR_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
        processorOptions: { batchSamples: FRAME_SAMPLES },
      });
      source.connect(node);
      node.connect(context.destination); // the worklet itself outputs nothing

      node.port.onmessage = (event) => {
        const data = event.data;
        if (!data.samples) return;
        if (socket.readyState !== WebSocket.OPEN) return;
        const samples = data.samples;
        // Float32 little-endian, exactly what both engines want.
        socket.send(samples.buffer.slice(0));
        state.samplesSent += samples.length;
        if (state.samplesSent / TARGET_SAMPLE_RATE >= MAX_SECONDS) {
          stop();
        }
      };

      state.context = context;
      state.stream = stream;
      state.source = source;
      state.node = node;
      state.phase = "live";
      state.startedAt = performance.now();
      refreshButtons();
      setStatus("Listening. Audio is being transcribed as you speak.", "live");
      if (hint) {
        hint.textContent =
          "Captured at " + TARGET_SAMPLE_RATE + " Hz mono; " +
          FRAME_SAMPLES / TARGET_SAMPLE_RATE * 1000 + " ms frames, " +
          control.cadence + " s per engine pass.";
      }
    }

    async function start() {
      if (state.phase === "live" || state.phase === "starting") {
        stop();
        return;
      }
      if (!window.isSecureContext) {
        setStatus(
          "Microphone access needs a secure context. Use https:// or http://127.0.0.1.",
          "bad"
        );
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("This browser does not expose microphone capture.", "bad");
        return;
      }

      state.phase = "starting";
      refreshButtons();
      setStatus("Requesting microphone access…", null);

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
      } catch (error) {
        state.phase = "idle";
        refreshButtons();
        setStatus(describeError(error), "bad");
        return;
      }

      try {
        await startWithStream(stream);
      } catch (error) {
        teardown();
        if (state.socket) {
          try {
            state.socket.close();
          } catch (ignored) {
            /* already closing */
          }
          state.socket = null;
        }
        state.phase = "idle";
        refreshButtons();
        setStatus(String(error.message || error), "bad");
      }
    }

    function stop() {
      if (state.phase !== "live") {
        return;
      }
      state.phase = "stopping";
      clearTimers();
      teardown();
      setStatus("Finishing the last phrases…", null);
      refreshButtons();
      if (state.socket && state.socket.readyState === WebSocket.OPEN) {
        state.socket.send(JSON.stringify({ type: "stop" }));
      } else {
        handleEnded({ transcripts: {}, metrics: {}, errors: {} });
      }
    }

    function clear() {
      if (state.socket) {
        try {
          state.socket.close();
        } catch (error) {
          /* already closing */
        }
        state.socket = null;
      }
      teardown();
      clearTimers();
      panelsHost.innerHTML = "";
      state.panels.clear();
      finalHost.innerHTML = "";
      panelsCard.hidden = true;
      finalCard.hidden = true;
      state.results = null;
      state.phase = "idle";
      state.providers = controlState().providers;
      if (hint) hint.textContent = defaultHint;
      saveButton.disabled = true;
      saveStatus.textContent = "";
      setStatus("Ready.", null);
      refreshButtons();
    }

    async function save() {
      if (!state.results) {
        return;
      }
      saveButton.disabled = true;
      saveStatus.textContent = "Saving…";
      const body = {
        providers: state.providers,
        transcripts: state.results.transcripts || {},
        metrics: state.results.metrics || {},
        errors: state.results.errors || {},
        words: state.results.words || {},
        durationMs: Math.round((state.samplesSent / TARGET_SAMPLE_RATE) * 1000),
        language: state.options ? state.options.language : null,
        keywords: state.options ? state.options.keywords : "",
        wordTimestamps: state.options ? state.options.wordTimestamps : false,
        cadenceSeconds: state.options ? state.options.cadence : null,
        referenceText: (document.getElementById("rt-reference") || {}).value || "",
        notes: (document.getElementById("rt-notes") || {}).value || "",
      };
      try {
        const response = await fetch("/api/realtime/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const payload = await response.json();
        if (payload.saved) {
          saveStatus.innerHTML =
            'Saved as <a href="/runs/' + payload.runId + '">' + payload.runId + "</a>.";
        } else {
          saveStatus.textContent = payload.message || "Could not save.";
          saveButton.disabled = false;
        }
      } catch (error) {
        saveStatus.textContent = "Could not save: " + error;
        saveButton.disabled = false;
      }
    }

    function copyAll() {
      if (!state.results) return;
      const text = Object.keys(state.results.transcripts || {})
        .map((name) => name + ":\n" + (state.results.transcripts[name] || ""))
        .join("\n\n");
      navigator.clipboard.writeText(text).then(
        () => {
          saveStatus.textContent = "Copied.";
        },
        () => {
          saveStatus.textContent = "Clipboard was blocked.";
        }
      );
    }

    recordButton.addEventListener("click", start);
    stopButton.addEventListener("click", stop);
    clearButton.addEventListener("click", clear);
    saveButton.addEventListener("click", save);
    copyButton.addEventListener("click", copyAll);
    document.querySelectorAll('input[name="rt-provider"]').forEach((box) => {
      box.addEventListener("change", refreshButtons);
    });
    window.addEventListener("pagehide", () => {
      if (state.phase === "live") teardown();
    });

    state.providers = controlState().providers;
    refreshButtons();
    setStatus(
      state.providers.length ? "Ready." : "No model selected.", null
    );

    return {
      start: start,
      stop: stop,
      clear: clear,
      startWithStream: startWithStream,
      handleMessage: handleMessage,
      getState: () => ({
        phase: state.phase,
        samplesSent: state.samplesSent,
        providers: state.providers,
        seconds: state.samplesSent / TARGET_SAMPLE_RATE,
        transcripts: state.results ? state.results.transcripts : null,
      }),
    };
  }

  const api = {
    TARGET_SAMPLE_RATE: TARGET_SAMPLE_RATE,
    FRAME_SAMPLES: FRAME_SAMPLES,
    createController: createController,
    controller: null,
  };

  function boot() {
    api.controller = createController();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }

  window.VoiceBenchRealtime = api;
})();
