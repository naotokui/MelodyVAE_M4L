// VAE in tensorflow.js
// based on https://github.com/songer1993/tfjs-vae

const Max = require('max-api');
const tf = require('@tensorflow/tfjs-node');

const utils = require('./utils.js')
const data = require('./data.js')

// Constants
const NUM_MIDI_CLASSES = require('./constants.js').NUM_MIDI_CLASSES;
const NUM_DRUM_CLASSES = require('./constants.js').NUM_DRUM_CLASSES;

const LOOP_DURATION = require('./constants.js').LOOP_DURATION;

const ORIGINAL_DIM = require('./constants.js').ORIGINAL_DIM;
const ORIGINAL_DIM_DRUMS = require('./constants.js').ORIGINAL_DIM_DRUMS;

const INTERMEDIATE_DIM = 512;
const LATENT_DIM = 2;

const BATCH_SIZE = 128;
const NUM_BATCH = 50;
const TEST_BATCH_SIZE = 1000;
const ON_LOSS_COEF = 0.75;  // coef for onsets loss
const DUR_LOSS_COEF = 1.0;  // coef for duration loss
const VEL_LOSS_COEF = 2.5;  // coef for velocity loss
const TS_LOSS_COEF = 5.0;  // coef for timeshift loss

let dataHandlerOnset;
let dataHandlerVelocity;
let dataHandlerDuration;
let dataHandlerOnsetDr;
let dataHandlerVelocityDr;
let dataHandlerTimeshiftDr;
let model;
let numEpochs = 100;

async function loadAndTrain(train_data_onset, train_data_velocity, train_data_duration,
    train_data_onset_dr, train_data_velocity_dr, train_data_timeshift_dr) {
  console.assert(train_data_onset.length == train_data_velocity.length 
    && train_data_velocity.length == train_data_duration.length 
    && train_data_duration.length == train_data_onset_dr.length);
  
  // shuffle in sync
  const total_num = train_data_onset.length;
  shuffled_indices = tf.util.createShuffledIndices(total_num);
  train_data_onset = utils.shuffle_with_indices(train_data_onset,shuffled_indices);
  train_data_velocity = utils.shuffle_with_indices(train_data_velocity,shuffled_indices);
  train_data_duration = utils.shuffle_with_indices(train_data_duration,shuffled_indices);
  train_data_onset_dr = utils.shuffle_with_indices(train_data_onset_dr,shuffled_indices);
  train_data_velocity_dr = utils.shuffle_with_indices(train_data_velocity_dr,shuffled_indices);
  train_data_timeshift_dr = utils.shuffle_with_indices(train_data_timeshift_dr,shuffled_indices);

  // synced indices
  const num_trains = Math.floor(data.TRAIN_TEST_RATIO * total_num);
  const num_tests  = total_num - num_trains;
  const train_indices = tf.util.createShuffledIndices(num_trains);
  const test_indices = tf.util.createShuffledIndices(num_tests);

  // create data handlers
  dataHandlerOnset = new data.DataHandler(train_data_onset, train_indices, test_indices, ORIGINAL_DIM); // data utility fo onset
  dataHandlerVelocity = new data.DataHandler(train_data_velocity, train_indices, test_indices, ORIGINAL_DIM); // data utility for velocity
  dataHandlerDuration = new data.DataHandler(train_data_duration, train_indices, test_indices, ORIGINAL_DIM); // data utility for duration
  dataHandlerOnsetDr = new data.DataHandler(train_data_onset_dr, train_indices, test_indices, ORIGINAL_DIM_DRUMS); // data utility fo onset
  dataHandlerVelocityDr = new data.DataHandler(train_data_velocity_dr, train_indices, test_indices, ORIGINAL_DIM_DRUMS); // data utility for velocity
  dataHandlerTimeshiftDr = new data.DataHandler(train_data_timeshift_dr, train_indices, test_indices, ORIGINAL_DIM_DRUMS); // data utility for duration

  // start training!
  initModel(); // initializing model class
  startTraining(); // start the actual training process with the given training data
}

function initModel(){
  model = new ConditionalVAE({
    modelConfig:{
      originalDim: ORIGINAL_DIM,
      originalDimDr: ORIGINAL_DIM_DRUMS,
      intermediateDim: INTERMEDIATE_DIM,
      latentDim: LATENT_DIM
    },
    trainConfig:{
      batchSize: 16,
      testBatchSize: TEST_BATCH_SIZE,
      // epochs: 50,
      optimizer: tf.train.adam(),
    //   logMessage: ui.logMessage,
    //   plotTrainLoss: ui.plotTrainLoss,
    //   plotValLoss: ui.plotValLoss,
    //   updateProgressBar: ui.updateProgressBar
    }
  });
}

async function startTraining(){
  await model.train();
}

function stopTraining(){
  model.shouldStopTraining = true;
  utils.log_status("Stopping training...");
}

function isTraining(){
  if (model && model.isTraining) return true;
}

function isReadyToGenerate(){
  // return (model && model.isTrained);
  return (model);
}

function setEpochs(e){
  numEpochs = e;
  Max.outlet("epoch", 0, numEpochs);
}

function generatePattern(z1, z2, noise_range=0.0){
  var zs;
  if (z1 === 'undefined' || z2 === 'undefined'){
    zs = tf.randomNormal([1, 2]);
  } else {
    zs = tf.tensor2d([[z1, z2]]);
  }

  // noise
  if (noise_range > 0.0){
    var noise = tf.randomNormal([1, 2]);
    zs = zs.add(noise.mul(tf.scalar(noise_range)));
  }
  return model.generate(zs);
}

async function saveModel(filepath){
  model.saveModel(filepath);
}

async function loadModel(filepath){
  if (!model) initModel();
  model.loadModel(filepath);
}

// Sampling Z 
class sampleLayer extends tf.layers.Layer {
  constructor(args) {
    super({});
  }

  computeOutputShape(inputShape) {
    return inputShape[0];
  }

  call(inputs, kwargs) {
    return tf.tidy(() => {
      const [zMean, zLogVar] = inputs;
      const batch = zMean.shape[0];
      const dim = zMean.shape[1];
      const epsilon = tf.randomNormal([batch, dim]);
      const half = tf.scalar(0.5);
      const temp = zLogVar.mul(half).exp().mul(epsilon);
      const sample = zMean.add(temp);
      return sample;
    });
  }

  getClassName() {
    return 'sampleLayer';
  }
}

  
class ConditionalVAE {
  constructor(config) {
    this.modelConfig = config.modelConfig;
    this.trainConfig = config.trainConfig;
    [this.encoder, this.decoder, this.apply] = this.build();
    this.isTrained = false;
  }

  build(modelConfig) {
    if (modelConfig != undefined){
      this.modelConfig = modelConfig;
    }
    const config = this.modelConfig;

    const originalDim = config.originalDim;
    const originalDimDr = config.originalDimDr;
    const intermediateDim = config.intermediateDim;
    const latentDim = config.latentDim;

    // VAE model = encoder + decoder
    // build encoder model

    // Onset Input
    const encoderInputsOn = tf.input({shape: [originalDim]});
    const x1LinearOn = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsOn);
    const x1NormalisedOn = tf.layers.batchNormalization({axis: 1}).apply(x1LinearOn);
    const x1On = tf.layers.leakyReLU().apply(x1NormalisedOn);

    // Velocity input
    const encoderInputsVel = tf.input({shape: [originalDim]});
    const x1LinearVel = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsVel);
    const x1NormalisedVel = tf.layers.batchNormalization({axis: 1}).apply(x1LinearVel);
    const x1Vel = tf.layers.leakyReLU().apply(x1NormalisedVel);
    
    // Duration input
    const encoderInputsDur = tf.input({shape: [originalDim]});
    const x1LinearDur = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsDur);
    const x1NormalisedDur = tf.layers.batchNormalization({axis: 1}).apply(x1LinearDur);
    const x1Dur = tf.layers.leakyReLU().apply(x1NormalisedDur);

    // Merged
    const concatLayer = tf.layers.concatenate();
    const x1Merged = concatLayer.apply([x1On, x1Vel, x1Dur]);
    const x2Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x1Merged);
    const x2Normalised = tf.layers.batchNormalization({axis: 1}).apply(x2Linear);
    const x2 = tf.layers.leakyReLU().apply(x2Normalised);

    // Onset Input Drum
    const encoderInputsOnDr = tf.input({shape: [originalDimDr]});
    const x1LinearOnDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsOnDr);
    const x1NormalisedOnDr = tf.layers.batchNormalization({axis: 1}).apply(x1LinearOnDr);
    const x1OnDr = tf.layers.leakyReLU().apply(x1NormalisedOnDr);

    // Velocity input Drum
    const encoderInputsVelDr = tf.input({shape: [originalDimDr]});
    const x1LinearVelDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsVelDr);
    const x1NormalisedVelDr = tf.layers.batchNormalization({axis: 1}).apply(x1LinearVelDr);
    const x1VelDr = tf.layers.leakyReLU().apply(x1NormalisedVelDr);
    
    // Timeshift input Drum
    const encoderInputsTsDr = tf.input({shape: [originalDimDr]});
    const x1LinearTsDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(encoderInputsTsDr);
    const x1NormalisedTsDr = tf.layers.batchNormalization({axis: 1}).apply(x1LinearTsDr);
    const x1TsDr = tf.layers.leakyReLU().apply(x1NormalisedTsDr);

    // Merged Drum
    const concatLayerDr = tf.layers.concatenate();
    const x1MergedDr = concatLayerDr.apply([x1OnDr, x1VelDr, x1TsDr]);
    const x2LinearDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x1MergedDr);
    const x2NormalisedDr = tf.layers.batchNormalization({axis: 1}).apply(x2LinearDr);
    const x2Dr = tf.layers.leakyReLU().apply(x2NormalisedDr);

    // Merged
    const concatLayer2 = tf.layers.concatenate();
    const x2Merged = concatLayer2.apply([x2, x2Dr]);
    const x3Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x2Merged);
    const x3Normalized = tf.layers.batchNormalization({axis: 1}).apply(x3Linear);
    const x3 = tf.layers.leakyReLU().apply(x3Normalized);
      
    const zMean = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const zLogVar = tf.layers.dense({units: latentDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x3);
    const z = new sampleLayer().apply([zMean, zLogVar]);

    const encoderInputs = [encoderInputsOn, encoderInputsVel, encoderInputsDur, 
                              encoderInputsOnDr, encoderInputsVelDr, encoderInputsTsDr];
    const encoderOutputs = [zMean, zLogVar, z];

    const encoder = tf.model({inputs: encoderInputs, outputs: encoderOutputs, name: "encoder"})

    // build decoder model
    const decoderInputs = tf.input({shape: [latentDim]});
    const x4Linear = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(decoderInputs);
    const x4Normalised = tf.layers.batchNormalization({axis: 1}).apply(x4Linear);
    const x4 = tf.layers.leakyReLU().apply(x4Normalised);

    // Decoder for onsets
    const x4LinearOn = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4);
    const x4NormalisedOn = tf.layers.batchNormalization({axis: 1}).apply(x4LinearOn);
    const x4On = tf.layers.leakyReLU().apply(x4NormalisedOn);
    const decoderOutputsOn = tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x4On);

    // Decoder for velocity
    const x4LinearVel = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4);
    const x4NormalisedVel = tf.layers.batchNormalization({axis: 1}).apply(x4LinearVel);
    const x4Vel = tf.layers.leakyReLU().apply(x4NormalisedVel);
    const decoderOutputsVel = tf.layers.dense({units: originalDim, activation: 'sigmoid'}).apply(x4Vel);
    
    // Decoder for duration
    const x4LinearDur = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4);
    const x4NormalisedDur = tf.layers.batchNormalization({axis: 1}).apply(x4LinearDur);
    const x4Dur = tf.layers.leakyReLU().apply(x4NormalisedDur);
    const decoderOutputsDur = tf.layers.dense({units: originalDim, activation: 'relu'}).apply(x4Dur);

    // build decoder model Drum
    const x4LinearDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(decoderInputs);
    const x4NormalisedDr = tf.layers.batchNormalization({axis: 1}).apply(x4LinearDr);
    const x4Dr = tf.layers.leakyReLU().apply(x4NormalisedDr);

    // Decoder for onsets Drum
    const x4LinearOnDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4Dr);
    const x4NormalisedOnDr = tf.layers.batchNormalization({axis: 1}).apply(x4LinearOnDr);
    const x4OnDr = tf.layers.leakyReLU().apply(x4NormalisedOnDr);
    const decoderOutputsOnDr = tf.layers.dense({units: originalDimDr, activation: 'sigmoid'}).apply(x4OnDr);

    // Decoder for velocity Drum
    const x4LinearVelDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4Dr);
    const x4NormalisedVelDr = tf.layers.batchNormalization({axis: 1}).apply(x4LinearVelDr);
    const x4VelDr = tf.layers.leakyReLU().apply(x4NormalisedVelDr);
    const decoderOutputsVelDr = tf.layers.dense({units: originalDimDr, activation: 'sigmoid'}).apply(x4VelDr);

    // Decoder for timeshift Drum
    const x4LinearTSDr = tf.layers.dense({units: intermediateDim, useBias: true, kernelInitializer: 'glorotNormal'}).apply(x4Dr);
    const x4NormalisedTSDr = tf.layers.batchNormalization({axis: 1}).apply(x4LinearTSDr);
    const x4TSDr = tf.layers.leakyReLU().apply(x4NormalisedTSDr);
    const decoderOutputsTSDr = tf.layers.dense({units: originalDimDr, activation: 'tanh'}).apply(x4TSDr);

    const decoderOutputs = [decoderOutputsOn, decoderOutputsVel, decoderOutputsDur, 
        decoderOutputsOnDr, decoderOutputsVelDr, decoderOutputsTSDr];

    // Decoder model
    const decoder = tf.model({inputs: decoderInputs, outputs: decoderOutputs, name: "decoder"})

    // build VAE model
    const vae = (inputs) => {
      return tf.tidy(() => {
        const [zMean, zLogVar, z] = this.encoder.apply(inputs);
        const outputs = this.decoder.apply(z);
        return [zMean, zLogVar, outputs];
      });
    }

    return [encoder, decoder, vae];
  }


  reconstructionLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let reconstruction_loss;
      reconstruction_loss = tf.metrics.binaryCrossentropy(yTrue, yPred)
      reconstruction_loss = reconstruction_loss.mul(tf.scalar(yPred.shape[1]));
      return reconstruction_loss;
    });
  }

  mseLoss(yTrue, yPred) {
    return tf.tidy(() => {
      let mse_loss = tf.metrics.meanSquaredError(yTrue, yPred);
      mse_loss = mse_loss.mul(tf.scalar(yPred.shape[1]));
      return mse_loss;
    });
  }


  klLoss(z_mean, z_log_var) {
    return tf.tidy(() => {
      let kl_loss;
      kl_loss = tf.scalar(1).add(z_log_var).sub(z_mean.square()).sub(z_log_var.exp());
      kl_loss = tf.sum(kl_loss, -1);
      kl_loss = kl_loss.mul(tf.scalar(-0.5));
      return kl_loss;
    });
  }

  vaeLoss(yTrue, yPred) {
    return tf.tidy(() => {
      const [yTrueOn, yTrueVel, yTrueDur, yTrueOnDr, yTrueVelDr, yTrueTsDr] = yTrue;
      const [z_mean, z_log_var, y] = yPred;
      const [yOn, yVel, yDur, yOnDr, yVelDr, yTsDr] = y;

      let onset_loss = this.reconstructionLoss(yTrueOn, yOn);
      onset_loss = onset_loss.mul(ON_LOSS_COEF);
      let velocity_loss = this.mseLoss(yTrueVel, yVel);
      velocity_loss = velocity_loss.mul(VEL_LOSS_COEF);
      let duration_loss = this.mseLoss(yTrueDur, yDur);
      duration_loss = duration_loss.mul(DUR_LOSS_COEF);

      let onset_loss_dr = this.reconstructionLoss(yTrueOnDr, yOnDr);
      onset_loss_dr = onset_loss_dr.mul(ON_LOSS_COEF);
      let velocity_loss_dr = this.mseLoss(yTrueVelDr, yVelDr);
      velocity_loss_dr = velocity_loss_dr.mul(VEL_LOSS_COEF);
      let timeshift_loss_dr = this.mseLoss(yTrueTsDr, yTsDr);
      timeshift_loss_dr = timeshift_loss_dr.mul(TS_LOSS_COEF);

      const kl_loss = this.klLoss(z_mean, z_log_var);
      // console.log("onset_loss", tf.mean(onset_loss).dataSync());
      // console.log("velocity_loss", tf.mean(velocity_loss).dataSync());
      // console.log("duration_loss",  tf.mean(duration_loss).dataSync());
      // console.log("kl_loss",  tf.mean(kl_loss).dataSync());
      const total_loss = tf.mean(onset_loss.add(velocity_loss).add(duration_loss)
                  .add(onset_loss_dr).add(velocity_loss_dr).add(timeshift_loss_dr).add(kl_loss)); // averaged in the batch
      return total_loss;
    });
  }

  async train(data, trainConfig) {
    this.isTrained = false;
    this.isTraining = true;
    this.shouldStopTraining = false;
    if (trainConfig != undefined){
      this.trainConfig = trainConfig;
    }
    const config = this.trainConfig;

    const batchSize = config.batchSize;
    const numBatch = Math.floor(dataHandlerOnset.getDataSize() / batchSize);
    const epochs = numEpochs;
    const testBatchSize = config.testBatchSize;
    const optimizer = config.optimizer;
    const logMessage = console.log;
    const plotTrainLoss = console.log;
    const plotValLoss = console.log;
  //   const updateProgressBar = config.updateProgressBar;

    const originalDim = this.modelConfig.originalDim;
    const originalDimDr = this.modelConfig.originalDimDr;

    Max.outlet("training", 1);
    for (let i = 0; i < epochs; i++) {
      if (this.shouldStopTraining) break;

      let batchInputOn, batchInputVel, batchInputDur;
      let batchInputOnDr, batchInputVelDr, batchInputTsDr;
      let trainLoss;
      let epochLoss;

      logMessage(`[Epoch ${i + 1}]\n`);
      Max.outlet("epoch", i + 1, epochs);
      utils.log_status(`Epoch: ${i + 1}`);

      epochLoss = 0;
      for (let j = 0; j < numBatch; j++) {
        batchInputOn = dataHandlerOnset.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputVel = dataHandlerVelocity.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputDur = dataHandlerDuration.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDim]);
        batchInputOnDr = dataHandlerOnsetDr.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDimDr]);
        batchInputVelDr = dataHandlerVelocityDr.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDimDr]);
        batchInputTsDr = dataHandlerTimeshiftDr.nextTrainBatch(batchSize).xs.reshape([batchSize, originalDimDr]);
        trainLoss = await optimizer.minimize(() => this.vaeLoss([batchInputOn, batchInputVel, batchInputDur, 
                    batchInputOnDr, batchInputVelDr, batchInputTsDr], 
            this.apply([batchInputOn, batchInputVel, batchInputDur, batchInputOnDr, batchInputVelDr, batchInputTsDr])), true);
        trainLoss = Number(trainLoss.dataSync());
        epochLoss = epochLoss + trainLoss;
        // logMessage(`\t[Batch ${j + 1}] Training Loss: ${trainLoss}.\n`);
        //plotTrainLoss(trainLoss);

        await tf.nextFrame();
      }
      epochLoss = epochLoss / numBatch;
      logMessage(`\t[Average] Training Loss: ${epochLoss}.\n`);
      logMessage(i, epochs);

      Max.outlet("loss", epochLoss);
      // testBatchInput = data.nextTrainBatch(testBatchSize).xs.reshape([testBatchSize, originalDim]);
      // testBatchResult = this.apply(testBatchInput);
      // valLoss = this.vaeLoss(testBatchInput, testBatchResult);
      // valLoss = Number(valLoss.dataSync());
      // plotValLoss(valLoss);
      await tf.nextFrame();
    }
    this.isTrained = true;
    this.isTraining = false;
    Max.outlet("training", 0);
    utils.log_status("Training finished!");
  }
  
  generate(zs){
    let [outputsOn, outputsVel, outputsDur] = this.decoder.apply(zs);

    outputsOn = outputsOn.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);     
    outputsVel = outputsVel.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);    
    outputsDur = outputsDur.reshape([NUM_MIDI_CLASSES, LOOP_DURATION]);    

    return [outputsOn.arraySync(), outputsVel.arraySync(), outputsDur.arraySync()];
  }

  async saveModel(path){
    const saved = await this.decoder.save(path);
    utils.post(saved);
  }

  async loadModel(path){
    this.decoder = await tf.loadLayersModel(path);
    this.isTrained = true;
  }
}

function range(start, edge, step) {
  // If only one number was passed in make it the edge and 0 the start.
  if (arguments.length == 1) {
    edge = start;
    start = 0;
  }

  // Validate the edge and step numbers.
  edge = edge || 0;
  step = step || 1;

  // Create the array of numbers, stopping befor the edge.
  for (var ret = []; (edge - start) * step > 0; start += step) {
    ret.push(start);
  }
  return ret;
}

exports.loadAndTrain = loadAndTrain;
exports.saveModel = saveModel;
exports.loadModel = loadModel;
exports.generatePattern = generatePattern;
exports.stopTraining = stopTraining;
exports.isReadyToGenerate = isReadyToGenerate;
exports.isTraining = isTraining;
exports.setEpochs = setEpochs;

