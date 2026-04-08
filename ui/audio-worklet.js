/**
 * AudioWorklet processor -- captures mic audio as PCM16 frames.
 *
 * Buffers 512 samples (32ms at 16kHz) to match Silero-VAD's
 * expected chunk size. Sends 1024-byte binary messages (512 x int16).
 *
 * The AudioContext is created with sampleRate: 16000, so the browser
 * handles hardware resampling internally. No decimation filter needed.
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(512);
    this._offset = 0;
    this._muted = false;

    this.port.onmessage = (event) => {
      if (event.data.type === 'mute') this._muted = true;
      if (event.data.type === 'unmute') this._muted = false;
    };
  }

  process(inputs) {
    if (this._muted) return true;

    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, 128 samples per call

    let i = 0;
    while (i < samples.length) {
      const remaining = 512 - this._offset;
      const toCopy = Math.min(remaining, samples.length - i);

      this._buffer.set(samples.subarray(i, i + toCopy), this._offset);
      this._offset += toCopy;
      i += toCopy;

      if (this._offset >= 512) {
        // Convert float32 to int16 PCM
        const pcm16 = new Int16Array(512);
        for (let j = 0; j < 512; j++) {
          const s = Math.max(-1, Math.min(1, this._buffer[j]));
          pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
        this._buffer = new Float32Array(512);
        this._offset = 0;
      }
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
