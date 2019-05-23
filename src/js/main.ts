import * as tf from '@tensorflow/tfjs';
import { parse } from './npy';

declare global {
  interface Window {
    pageLoaded?: Function;
    transcribe?: Function;
  }
}

const BLANK_INDEX: number = 28;
const CHAR_MAP: string = " abcdefghijklmnopqrstuvwxyz'-";
const SAMPLE_RATE = 16000;
const WINDOW_SIZE = 20; // 20 mS
const STRIDE_SIZE = 10; // 10 mS
let statusElement: HTMLLabelElement;

class DeepSpeech {
  model: tf.GraphModel;

  // load the tfjs-graph model
  async load() {
    this.model = await tf.loadGraphModel('/model/model.json');
  }

  // CTC greedy decoder implementation
  _decodeStr(source: number[]) : string {
    return source.reduce( (pre, curr, ) => {
      // groupby
      pre = pre || [];
      if (pre[pre.length - 1] !== curr) {
        pre.push(curr);
      }
      return pre;
    }, [])
    .filter( (item) => {
      // remove blank index
      return item !== BLANK_INDEX;
    }).map((charIndex) => {
      // map to char
      return CHAR_MAP.charAt(charIndex);
    }).join('');
  }

  async transcribe(url: string) {
    let feature = await prepareInput(url);
    let pred: tf.Tensor[] = await this.model.executeAsync(
        {'features': feature.expandDims(0)}) as tf.Tensor[];
    feature.dispose();
    // pred[0] == argmax of pred[1] against axes=2
    // so directly use pred[0] and cleanup pred[1]
    let t = tf.tidy(() => {
      return pred[0].squeeze([0]);
    });
    pred[1].dispose();
    let argmax = t.arraySync() as number[];
    t.dispose();
    return this._decodeStr(argmax);
  }
}

// class AudioProcessor {

// }

const ds = new DeepSpeech();

async function pageLoaded() {
  let start = Date.now();
  await measureElapsedTime('loading model', async () => {
    await loadModel();
  });
  // await measureElapsedTime('transcribe', async () => {
  //   await transcribe('/npy/14-feature.npy');
  // });
}

async function measureElapsedTime(task: string, func: Function) {
  const start = Date.now();
  await func();
  console.log(`${task}: ${Date.now() - start}`);
}

// Load the model and update UI
async function loadModel() {
  updateStatus('loading DS2 model.....');
  await ds.load();
  updateStatus('DS2 model loaded!');
}

async function stft(buff: Float32Array) {
  const windowSize = SAMPLE_RATE * 0.001 * WINDOW_SIZE;
  const strideSize = SAMPLE_RATE * 0.001 * STRIDE_SIZE;
  const trancate = (buff.length - windowSize ) % strideSize;
  let trancatedBuff = new Float32Array(
      buff.buffer, 0, buff.length - trancate);
  let stridedBuff = getStrideBuff(trancatedBuff, windowSize, strideSize);
  console.log(`buff.length: ${buff.buffer.byteLength}, trancated: ${buff.length - trancate}`);
  let shape = [(trancatedBuff.length - windowSize)/strideSize + 1, windowSize];
  console.log(shape);
  let fft = tf.tidy(() => {
    let hannWin = tf.hannWindow(windowSize).expandDims(0);
    let windows = tf.tensor(stridedBuff, shape, 'float32');
    let fft = windows.mul(hannWin).rfft().abs().square();
    // let scale = hannWin.square().sum().mul(tf.scalar(SAMPLE_RATE));
    return fft;
  });
  console.log(fft.shape);
  console.log(fft.arraySync());
}

function getStrideBuff(buff: Float32Array, window: number, stride: number): Float32Array {
  let shape = [ window, (buff.length - window)/stride + 1];
  let rev = new Float32Array(shape[0] * shape[1]);
  for( let i = 0; i < shape[1]; i++) {
    let tmp = new Float32Array(buff.buffer, (i * stride) * 4, window);
    rev.set(tmp, i * window);
  }
  return rev;
}

// Transcribe and update UI
async function transcribe() {
  const fileInput = document.getElementById('audio-file') as HTMLInputElement;
  // using the sampling rate while training
  const audioCtx = new AudioContext({sampleRate: SAMPLE_RATE});
  const fr = new FileReader();
  fr.onload = (ev: ProgressEvent) => {
    audioCtx.decodeAudioData(fr.result as ArrayBuffer).then( async (buff: AudioBuffer) => {
      updateStatus(`sample rate:${buff.sampleRate}, ch #: ${buff.numberOfChannels}, length:${buff.getChannelData(0).length}`);
      await stft(buff.getChannelData(0));
    })
  };
  fr.readAsArrayBuffer(fileInput.files[0]);
}
// async function transcribe(url: string) {
//   updateStatus('transcribe.....');
//   let transcription = await ds.transcribe(url);
//   updateStatus(`transcription: ${transcription}`);
// }

function prepareInput(url: string): Promise<tf.Tensor> {
  return new Promise((resolve, reject) => {
    fetch(url)
    .then((response) => response.arrayBuffer())
    .then((data) => resolve(parse(data)));
  });
}

function updateStatus(msg: string) {
  if (!statusElement) {
    statusElement = document.getElementById('status') as HTMLLabelElement;
  }
  statusElement.innerHTML = msg;
}

// attach functions to window object to prevent the pruning
window.pageLoaded = pageLoaded;
window.transcribe = transcribe;