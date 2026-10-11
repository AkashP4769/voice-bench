// AudioWorklet that collects microphone samples and hands them to the page.
//
// Samples are batched and transferred rather than posted one quantum at a time:
// an AudioWorklet quanta is 128 frames, so per-quantum messages would be ~375
// small posts per second for no benefit.
//
// This runs on the audio render thread, so it must not allocate more than it
// has to and must never touch the DOM.

"use strict";

const DEFAULT_BATCH_SAMPLES = 4096;

class VoiceBenchRecorder extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Realtime wants smaller frames for lower latency; the batch recorder keeps
    // its larger default.
    const requested =
      options && options.processorOptions && options.processorOptions.batchSamples;
    this._batch = Number.isFinite(requested) && requested > 0
      ? Math.floor(requested)
      : DEFAULT_BATCH_SAMPLES;
    this._buffer = new Float32Array(this._batch);
    this._offset = 0;
    this._peak = 0;
    this._closed = false;
    this.port.onmessage = (event) => {
      if (event.data === "flush") {
        this._flush();
        // Sent after any sample messages, and port messages are ordered, so
        // the page can treat this as "everything captured so far has arrived".
        this.port.postMessage({ flushed: true });
      }
    };
  }

  process(inputs) {
    const channels = inputs[0];
    if (this._closed || !channels || channels.length === 0) {
      return true;
    }

    const input = channels[0];
    for (let index = 0; index < input.length; index += 1) {
      const sample = input[index];
      this._buffer[this._offset] = sample;
      this._offset += 1;

      const magnitude = sample < 0 ? -sample : sample;
      if (magnitude > this._peak) {
        this._peak = magnitude;
      }

      if (this._offset === this._batch) {
        this._flush();
      }
    }
    return true;
  }

  _flush() {
    if (this._offset === 0) {
      return;
    }
    // Copy the filled portion, then transfer that copy's buffer: transferring
    // the batch buffer itself would detach it while process() still uses it.
    const samples = this._buffer.slice(0, this._offset);
    this.port.postMessage({ samples, peak: this._peak }, [samples.buffer]);
    this._buffer = new Float32Array(this._batch);
    this._offset = 0;
    this._peak = 0;
  }
}

registerProcessor("voicebench-recorder", VoiceBenchRecorder);
