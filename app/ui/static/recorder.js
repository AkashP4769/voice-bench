// Microphone recording for the benchmark form.
//
// Why not MediaRecorder: it produces webm/opus (or mp4/aac), and the backend
// decodes audio with libsndfile, which reads wav, flac, ogg and aiff but not
// webm or mp4. Fermion's own server rejects mp3 and m4a too. So raw PCM is
// captured through the Web Audio graph and encoded to 16 kHz mono WAV here,
// which is the format both engines natively want.
//
// The finished recording is placed on the existing file input rather than being
// posted separately, so the form keeps submitting every other field (providers,
// language, options, reference text) exactly as the user configured it.
//
// Requires a secure context: getUserMedia is unavailable on plain http:// except
// on localhost, which this app uses.

(function () {
  "use strict";

  const TARGET_SAMPLE_RATE = 16000;
  const PROCESSOR_NAME = "voicebench-recorder";
  const MAX_SECONDS = 300;
  // Below this, a recording is almost certainly a mis-tap rather than speech.
  const MIN_USEFUL_SECONDS = 0.3;

  // The worklet is fetched by URL, so it has to be versioned by hand: a cached
  // copy alongside updated page code is a confusing mismatch.
  const SCRIPT_VERSION = (function () {
    const tag = document.currentScript;
    return tag && tag.dataset ? tag.dataset.version || "" : "";
  })();
  const WORKLET_URL =
    "/static/recorder-worklet.js" + (SCRIPT_VERSION ? "?v=" + SCRIPT_VERSION : "");

  // ---------------------------------------------------------------- helpers

  function mergeChunks(chunks) {
    let total = 0;
    for (const chunk of chunks) {
      total += chunk.length;
    }
    const merged = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  async function resampleTo16k(chunks, inputRate) {
    const merged = mergeChunks(chunks);
    if (merged.length === 0) {
      return merged;
    }
    if (inputRate === TARGET_SAMPLE_RATE) {
      return merged;
    }
    // OfflineAudioContext is the browser's own resampler, so the client and the
    // server do not disagree about how to downsample.
    const frames = Math.max(1, Math.round((merged.length * TARGET_SAMPLE_RATE) / inputRate));
    const offline = new OfflineAudioContext(1, frames, TARGET_SAMPLE_RATE);
    const source = offline.createBuffer(1, merged.length, inputRate);
    source.copyToChannel(merged, 0);
    const node = offline.createBufferSource();
    node.buffer = source;
    node.connect(offline.destination);
    node.start();
    const rendered = await offline.startRendering();
    return rendered.getChannelData(0);
  }

  function writeAscii(view, offset, text) {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index));
    }
  }

  function encodeWav(samples, sampleRate) {
    const bytesPerSample = 2;
    const dataBytes = samples.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataBytes);
    const view = new DataView(buffer);

    writeAscii(view, 0, "RIFF");
    view.setUint32(4, 36 + dataBytes, true);
    writeAscii(view, 8, "WAVE");
    writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true); // PCM header size
    view.setUint16(20, 1, true); // format: PCM
    view.setUint16(22, 1, true); // channels: mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
    view.setUint16(32, bytesPerSample, true); // block align
    view.setUint16(34, 8 * bytesPerSample, true); // bits per sample
    writeAscii(view, 36, "data");
    view.setUint32(40, dataBytes, true);

    let offset = 44;
    for (let index = 0; index < samples.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, samples[index]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
    return new Blob([view], { type: "audio/wav" });
  }

  function formatSeconds(seconds) {
    if (seconds < 60) {
      return seconds.toFixed(1) + " s";
    }
    const whole = Math.floor(seconds / 60);
    const rest = seconds - whole * 60;
    return whole + "m " + rest.toFixed(1).padStart(4, "0") + "s";
  }

  function recordingName() {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    return "recording-" + stamp + ".wav";
  }

  // ------------------------------------------------------------- controller

  function createController() {
    const fileInput = document.getElementById("audio-file");
    const fileLabel = document.getElementById("audio-file-label");
    const recordButton = document.getElementById("record-button");
    const stopButton = document.getElementById("record-stop");
    const clearButton = document.getElementById("record-clear");
    const panel = document.getElementById("recorder");
    const status = document.getElementById("audio-status");
    const timer = document.getElementById("record-timer");
    const levelFill = document.getElementById("record-level-fill");
    const hint = document.getElementById("record-hint");
    const submitButton = document.getElementById("run-button");

    if (!fileInput || !recordButton || !panel) {
      return null;
    }

    const DEFAULT_HINT = hint ? hint.textContent : "";

    const state = {
      phase: "idle", // idle | recording | processing | ready
      context: null,
      stream: null,
      node: null,
      source: null,
      chunks: [],
      sampleRate: TARGET_SAMPLE_RATE,
      startedAt: 0,
      elapsed: 0,
      displayPeak: 0,
      ticks: null,
      objectUrl: null,
      recording: null, // { file, seconds }
    };

    function setStatus(message, tone) {
      status.textContent = message;
      status.className = "source-status" + (tone ? " " + tone : "");
    }

    function updateLevel(peak) {
      state.displayPeak = Math.max(peak, state.displayPeak * 0.86);
      if (levelFill) {
        levelFill.style.width = Math.min(100, state.displayPeak * 140).toFixed(1) + "%";
      }
    }

    function resetLevel() {
      state.displayPeak = 0;
      if (levelFill) {
        levelFill.style.width = "0%";
      }
    }

    function refreshSubmitState() {
      if (submitButton) {
        submitButton.disabled = state.phase === "recording" || state.phase === "processing";
      }
    }

    function showIdle(detail) {
      panel.hidden = true;
      resetLevel();
      state.phase = "idle";
      recordButton.classList.remove("recording");
      recordButton.setAttribute("aria-pressed", "false");
      recordButton.title = "Record from microphone";
      if (hint) {
        hint.textContent = DEFAULT_HINT;
      }
      refreshSubmitState();
      if (detail) {
        setStatus(detail, null);
      }
    }

    function discard() {
      // Abandon whatever is in progress or staged, releasing the microphone on
      // the way out. Safe to call in any phase: teardown tolerates nulls, and
      // setting the phase to idle makes a late finalize() return early.
      teardown();
      forgetRecording();
      clearFileInput();
      if (fileLabel) {
        fileLabel.classList.remove("has-input");
      }
      if (timer) {
        timer.textContent = "0.0 s";
      }
      showIdle("No audio selected.");
    }

    function forgetRecording() {
      state.recording = null;
      if (state.objectUrl) {
        URL.revokeObjectURL(state.objectUrl);
        state.objectUrl = null;
      }
    }

    function clearFileInput() {
      fileInput.value = "";
    }

    function attachRecording(blob, seconds) {
      if (typeof DataTransfer === "undefined") {
        throw new Error(
          "this browser cannot attach the recording to the form; save the audio " +
            "and use Choose audio file instead"
        );
      }
      const file = new File([blob], recordingName(), { type: "audio/wav" });
      const transfer = new DataTransfer();
      transfer.items.add(file);
      // Assigning files lets the existing form — and HTMX — carry the recording
      // as a normal upload, with every other field untouched.
      fileInput.files = transfer.files;
      forgetRecording();
      state.recording = { file: file, seconds: seconds };
      state.phase = "ready";
      refreshSubmitState();

      // The panel stays visible, but now reports the finished take rather than a
      // running one: same place in the layout, no second set of elements.
      panel.hidden = false;
      panel.dataset.phase = "ready";
      if (timer) {
        timer.textContent = formatSeconds(seconds);
      }

      recordButton.classList.remove("recording");
      recordButton.setAttribute("aria-pressed", "false");
      recordButton.title = "Record again";

      // The clip is still attached, because a deliberately short recording is
      // legitimate; the warning just explains why the transcript may be empty.
      const tooShort = seconds < MIN_USEFUL_SECONDS;
      setStatus(
        "Recording ready: " + formatSeconds(seconds) + " of 16 kHz mono audio (" +
          file.name + "). " +
          (tooShort
            ? "That is very short, so it may transcribe to nothing. Record again if " +
              "you meant to capture speech."
            : "Press Run benchmark to transcribe it."),
        tooShort ? "warn" : "ok"
      );
      if (fileLabel) {
        fileLabel.classList.add("has-input");
      }
    }

    function teardown() {
      if (state.ticks) {
        clearInterval(state.ticks);
        state.ticks = null;
      }
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
        // Releasing the tracks is what turns the browser's recording indicator off.
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
      }
      if (state.context) {
        const context = state.context;
        state.context = null;
        context.close().catch(() => {});
      }
    }

    function finishCapture() {
      const captured = state.chunks.slice();
      const rate = state.sampleRate;
      state.chunks = [];
      teardown();
      return { captured: captured, rate: rate };
    }

    function fail(error) {
      teardown();
      clearFileInput();
      forgetRecording();
      showIdle(null);
      setStatus(describeError(error), "bad");
    }

    function describeError(error) {
      const name = error && error.name ? error.name : "";
      if (name === "NotAllowedError" || name === "SecurityError") {
        return (
          "Microphone access was blocked. Allow it for this page in the browser's " +
          "site permissions, then record again."
        );
      }
      if (name === "NotFoundError" || name === "OverconstrainedError") {
        return "No microphone was found. Connect one, or choose an audio file instead.";
      }
      if (name === "NotReadableError") {
        return (
          "The microphone is in use by another application. Close it there, then " +
          "record again."
        );
      }
      return "Recording failed: " + (error && error.message ? error.message : String(error));
    }

    async function startWithStream(stream) {
      const context = new AudioContext();
      try {
        if (context.state === "suspended") {
          await context.resume();
        }
        await context.audioWorklet.addModule(WORKLET_URL);
        const source = context.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(context, PROCESSOR_NAME, {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          channelCountMode: "explicit",
          channelInterpretation: "speakers",
        });

        // The microphone goes straight into the worklet. A gain node was
        // previously inserted here to "mute the passthrough", which zeroed the
        // audio the worklet received: the recording came out as pure silence.
        // It is unnecessary anyway, because the worklet writes nothing to its
        // output, so routing that output to the destination is already silent.
        source.connect(node);
        node.connect(context.destination);

        node.port.onmessage = (event) => {
          const data = event.data;
          if (data.samples) {
            state.chunks.push(data.samples);
            updateLevel(data.peak || 0);
          }
        };

        state.context = context;
        state.stream = stream;
        state.source = source;
        state.node = node;
        state.sampleRate = context.sampleRate;
        state.chunks = [];
        state.startedAt = performance.now();
        state.elapsed = 0;
        state.phase = "recording";

        panel.hidden = false;
        panel.dataset.phase = "recording";
        recordButton.classList.add("recording");
        recordButton.setAttribute("aria-pressed", "true");
        recordButton.title = "Stop recording";
        if (hint) {
          hint.textContent =
            "Captured at " + context.sampleRate + " Hz and converted to 16 kHz mono on stop.";
        }
        refreshSubmitState();
        setStatus("Recording… press stop when finished.", "live");
        resetLevel();

        state.ticks = setInterval(() => {
          state.elapsed = (performance.now() - state.startedAt) / 1000;
          if (timer) {
            timer.textContent = formatSeconds(state.elapsed);
          }
          if (state.elapsed >= MAX_SECONDS) {
            stop();
          }
        }, 100);
      } catch (error) {
        try {
          stream.getTracks().forEach((track) => track.stop());
        } catch (ignored) {
          /* nothing to release */
        }
        try {
          await context.close();
        } catch (ignored) {
          /* already closed */
        }
        throw error;
      }
    }

    async function start() {
      if (state.phase === "recording") {
        stop();
        return;
      }
      if (state.phase === "processing") {
        return;
      }
      if (!window.isSecureContext) {
        setStatus(
          "Microphone access needs a secure context. Open the app over https:// " +
            "or on http://127.0.0.1, or choose an audio file instead.",
          "bad"
        );
        return;
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus(
          "This browser does not expose microphone capture. Choose an audio file instead.",
          "bad"
        );
        return;
      }

      clearFileInput();
      forgetRecording();
      state.phase = "processing";
      refreshSubmitState();
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
        fail(error);
        return;
      }

      try {
        await startWithStream(stream);
      } catch (error) {
        fail(error);
      }
    }

    function stop() {
      if (state.phase !== "recording") {
        return;
      }
      state.phase = "processing";
      if (state.ticks) {
        clearInterval(state.ticks);
        state.ticks = null;
      }
      state.elapsed = (performance.now() - state.startedAt) / 1000;
      refreshSubmitState();

      const node = state.node;
      if (!node) {
        finalize(false);
        return;
      }

      // Ask the worklet to flush its partial batch. It replies once flushed, and
      // port messages are ordered, so every sample message has arrived by then.
      node.port.onmessage = (event) => {
        const data = event.data;
        if (data.samples) {
          state.chunks.push(data.samples);
        } else if (data.flushed) {
          finalize(true);
        }
      };
      try {
        node.port.postMessage("flush");
      } catch (error) {
        finalize(false);
        return;
      }
      // Do not let a lost reply strand the UI in the processing state.
      setTimeout(() => {
        if (state.phase === "processing") {
          finalize(false);
        }
      }, 1500);
    }

    async function finalize() {
      if (state.phase !== "processing") {
        return;
      }
      const captured = finishCapture();
      setStatus("Preparing the recording…", null);

      try {
        if (captured.captured.length === 0) {
          throw new Error("the recording contained no audio");
        }
        const samples = await resampleTo16k(captured.captured, captured.rate);
        if (samples.length === 0) {
          throw new Error("the recording contained no audio");
        }
        const seconds = samples.length / TARGET_SAMPLE_RATE;
        const blob = encodeWav(samples, TARGET_SAMPLE_RATE);
        attachRecording(blob, seconds);
      } catch (error) {
        fail(error);
      }
    }

    function clear() {
      discard();
    }

    // A file chosen from disk replaces any recording.
    fileInput.addEventListener("change", () => {
      if (state.phase === "recording") {
        stop();
        return;
      }
      forgetRecording();
      panel.hidden = true;
      resetLevel();
      state.phase = "idle";
      recordButton.classList.remove("recording");
      recordButton.setAttribute("aria-pressed", "false");
      recordButton.title = "Record from microphone";
      refreshSubmitState();

      const file = fileInput.files && fileInput.files[0];
      if (file) {
        if (fileLabel) {
          fileLabel.classList.add("has-input");
        }
        setStatus("Selected " + file.name + " (" + formatBytes(file.size) + ").", "ok");
      } else {
        if (fileLabel) {
          fileLabel.classList.remove("has-input");
        }
        setStatus("No audio selected.", null);
      }
    });

    function formatBytes(bytes) {
      if (bytes < 1024) {
        return bytes + " B";
      }
      if (bytes < 1024 * 1024) {
        return (bytes / 1024).toFixed(1) + " KB";
      }
      return (bytes / 1024 / 1024).toFixed(1) + " MB";
    }

    recordButton.addEventListener("click", start);
    if (stopButton) {
      stopButton.addEventListener("click", stop);
    }
    if (clearButton) {
      clearButton.addEventListener("click", clear);
    }
    // A recording must not be left running if the page is navigated away.
    window.addEventListener("pagehide", () => {
      if (state.phase === "recording") {
        teardown();
      }
    });

    refreshSubmitState();
    return {
      start: start,
      stop: stop,
      clear: clear,
      startWithStream: startWithStream,
      finalize: finalize,
      getState: () => ({
        phase: state.phase,
        elapsed: state.elapsed,
        file: state.recording ? state.recording.file.name : null,
        seconds: state.recording ? state.recording.seconds : null,
      }),
    };
  }

  const api = {
    TARGET_SAMPLE_RATE: TARGET_SAMPLE_RATE,
    MAX_SECONDS: MAX_SECONDS,
    mergeChunks: mergeChunks,
    resampleTo16k: resampleTo16k,
    encodeWav: encodeWav,
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

  window.VoiceBenchRecorder = api;
})();
