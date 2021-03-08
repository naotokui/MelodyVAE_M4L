const path = require('path');
const Max = require('max-api');
const fs = require('fs')
const glob = require('glob');
const tf = require('@tensorflow/tfjs-node');
const { Midi } = require('@tonejs/midi'); // https://github.com/Tonejs/Midi

// Constants 
const MIN_MIDI_NOTE = require('./src/constants.js').MIN_MIDI_NOTE;
const MAX_MIDI_NOTE = require('./src/constants.js').MAX_MIDI_NOTE;
const NUM_MIDI_CLASSES = require('./src/constants.js').NUM_MIDI_CLASSES;
const LOOP_DURATION = require('./src/constants.js').LOOP_DURATION;
const MIN_ONSETS_THRESHOLD = require('./src/constants.js').MIN_ONSETS_THRESHOLD;
const INSTRUMENTS_MELODY = require('./src/constants.js').INSTRUMENTS_MELODY;

// VAE model and Utilities
const utils = require('./src/utils.js');
const vae = require('./src/vae.js');

// This will be printed directly to the Max console
Max.post(`Loaded the ${path.basename(__filename)} script`);

//outlet stored this message when this script loaded
Max.outlet("loaded");

// Global varibles
var train_data_onsets = [];
var train_data_velocities = []; 
var isGenerating = false;

function isValidMIDIFile(midiFile){
    if (midiFile.header.tempos.length > 1){
        utils.error("not compatible with midi files containing multiple tempo changes")
        return false;
    }
    return true;
}

function getTempo(midiFile){
    if (midiFile.header.tempos.length == 0) return 120.0 // no tempo info, then use 120.0 
    return midiFile.header.tempos[0].bpm;  // use the first tempo info and ignore tempo changes in MIDI file
}

// Get location of a note in pianoroll
function getNoteIndexAndTimeshift(note, tempo){
    const unit = (60.0 / tempo) / 4.0; // the duration of 16th note
    const half_unit = unit * 0.5;

    const index = Math.max(0, Math.floor((note.time + half_unit) / unit)) // with centering 
    const end_index = Math.max(0, Math.floor((note.time + note.duration + half_unit) / unit)) 
    
    return [index, end_index];
}

function getNumOfOnsets(onsets){
    var count = 0;
    for (var i = 0; i < NUM_MIDI_CLASSES; i++){
        for (var j=0; j < LOOP_DURATION; j++){
            if (onsets[i][j] > 0) count += 1;
        }
    }
    return count;
}

// Convert midi into pianoroll matrix
function processPianoroll(midiFile, augmentation, melody_only=false){
    const tempo = getTempo(midiFile);

    // data array
    var onsets = [];
    var velocities = [];

    midiFile.tracks.forEach(track => {
        if (track.channel != 9 && track.channel != 10){ // ignore drum tracks
            // ignore if the track is not one of the melody instruments
            let inst = track.instrument.number;
            console.log(inst);
            if (!melody_only || INSTRUMENTS_MELODY.includes(inst)){
                //notes are an array
                const notes = track.notes
                notes.forEach(note => {
                    // round pitch value in 0 - 24
                    let pitch = note.midi % 24;  // TODO: 
                    if (pitch < NUM_MIDI_CLASSES){
                        let timing = getNoteIndexAndTimeshift(note, tempo);
                        let index = timing[0];
                        let end_index = timing[1];  // index of note off

                        // add new array
                        while (Math.floor(end_index / LOOP_DURATION) >= onsets.length){
                            onsets.push(utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION));
                            velocities.push(utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION));
                        }

                        let note_id = pitch;

                        // store onset
                        for (let k = index; k <= end_index; k++){
                            let matrix = onsets[Math.floor(k / LOOP_DURATION)];
                            matrix[note_id][k % LOOP_DURATION] = 1;  // 1 for onsets   

                            // store velocity
                            matrix = velocities[Math.floor(k / LOOP_DURATION)];
                            matrix[note_id][k % LOOP_DURATION] = note.velocity;    // normalized 0 - 1
                        }  
                    } else {
                        console.log("out of scope ", note.midi);
                    }
                });
            }
        }
    })

    //data augmentation - with all keys
    if (augmentation){
        aug_onsets = [];
        aug_velocities = [];
        aug_durations = [];

        onsets.forEach(function (onset, i){
            let velocity = velocities[i];
            let maxv = utils.getMaxPitch(onset) + MIN_MIDI_NOTE;
            let minv = utils.getMinPitch(onset) + MIN_MIDI_NOTE;
            for (let diff = -12; diff <= 12; diff++){
                if (maxv + diff <= MAX_MIDI_NOTE && minv + diff >= MIN_MIDI_NOTE){ // if it's in the transposition range...
                    let newonset     = utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION);
                    let newvelocity = utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION);
                    for (var i = 0; i < NUM_MIDI_CLASSES; i++){
                        for (var j =0; j < LOOP_DURATION; j++){
                            if (i + diff >= 0 && i + diff < NUM_MIDI_CLASSES){
                                if (onset[i][j] > 0) {  // only if there is onset
                                    newonset[i + diff][j] = 1; // transpose
                                    newvelocity[i + diff][j] = velocity[i][j];
                                }
                            }
                        }
                    }
                    aug_onsets.push(newonset);                    
                    aug_velocities.push(newvelocity);
                }
            }
        });

        onsets.push(...aug_onsets);
        velocities.push(...aug_velocities);
    }

    console.assert(onsets.length == velocities.length,
         "Something wrong with augmentation? array length must be the same.");
    // /*    for debug - output pianoroll */
    // if (durations.length > 0){ 
    //     var index = utils.getRandomInt(durations.length); 
    //     let x = durations[index];
    //     // for (var i=0; i< NUM_MIDI_CLASSES; i++){
    //     //     for (var j=0; j < LOOP_DURATION; j++){
    //     //         // Max.outlet("matrix_output", j, i, Math.ceil(x[i][j]));
    //     //     }
    //     // }
    //     console.log(x);
    // }
    
    // 2D array to tf.tensor2d
    for (var i=0; i < onsets.length; i++){
        if (getNumOfOnsets(onsets[i]) > MIN_ONSETS_THRESHOLD){
            train_data_onsets.push(tf.tensor2d(onsets[i], [NUM_MIDI_CLASSES, LOOP_DURATION]));
            train_data_velocities.push(tf.tensor2d(velocities[i], [NUM_MIDI_CLASSES, LOOP_DURATION]));
        }
    }
}

function processMidiFile(filename, augmentation, melody_only){
    // // Read MIDI file into a buffer
    var input = fs.readFileSync(filename)

    var midiFile = new Midi(input);  
    if (isValidMIDIFile(midiFile) == false){
        utils.error("Invalid MIDI file: " + filename);
        return false;
    }

    var tempo = getTempo(midiFile);
    // console.log("tempo:", tempo);
    // console.log("signature:", midiFile.header.timeSignatures);
    processPianoroll(midiFile, augmentation, melody_only);
    console.log("processed:", filename);
    return true;
}

// Add training data
Max.addHandler("midi", (filename, augmentation, melody_only = false) =>  {
    var count = 0;
    // is directory? 
    if (fs.existsSync(filename) && fs.lstatSync(filename).isDirectory()){
        // iterate over *.mid or *.midi files 
        // TODO: it may match *.mido *.midifile *.middleageman etc...
        glob(filename + '**/*.mid', {}, (err, files)=>{
            if (err) console.log(err); 
            else {
                for (var idx in files){               
                    try {
                        if (processMidiFile(files[idx], augmentation, melody_only)) count += 1;
                    } catch(error) {
                        utils.error("failed to process " + files[idx] + " - " + error);
                    }
                }
                utils.post("# of midi files added: " + count);    
                reportNumberOfBars();
            }
        })
    } else {
        if (processMidiFile(filename, augmentation, melody_only)) count += 1;
        Max.post("# of midi files added: " + count);    
        reportNumberOfBars();
    }
});

// Start training! 
Max.addHandler("train", ()=>{
    if (vae.isTraining()){
        utils.error_status("Failed to start training. There is already an ongoing training process.");
        return;
    }

    utils.log_status("Start training...");
    console.log("# of bars in training data:", train_data_onsets.length * 2);
    reportNumberOfBars();
    vae.loadAndTrain(train_data_onsets, train_data_velocities);
});

// Generate a rhythm pattern
Max.addHandler("generate", (z1, z2, thresh_min, thresh_max = 1.0, noise_range = 0.0)=>{
    try {
        generatePattern(z1, z2, thresh_min, thresh_max, noise_range);
    } catch(error) {
        error_status(error);
    }
});

async function generatePattern(z1, z2, thresh_min, thresh_max, noise_range){
    if (vae.isReadyToGenerate()){    
      if (isGenerating) return;
  
      isGenerating = true;
      let [onsets, velocities] = vae.generatePattern(z1, z2, noise_range);
      Max.outlet("matrix_clear",1); // clear all
        
      // consolidate note ons/offs
      let true_onsets = utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION);
      let true_durs   = utils.create2DArray(NUM_MIDI_CLASSES, LOOP_DURATION);

      // For Grid
      for (var i=0; i< NUM_MIDI_CLASSES; i++){
          var sequence = [];
          // output for matrix view
          for (var j=0; j < LOOP_DURATION; j++){
              var x = 0.0;
              // if (pattern[i * LOOP_DURATION + j] > 0.2) x = 1;
              if (onsets[i][j] >= thresh_min && onsets[i][j] <= thresh_max){ 
                x = 1;
                Max.outlet("matrix_output", j + 1, i + 1, x); // index for live.grid starts from 1
              }
        } 

        // consolidating true note ons
        for (var j=0; j < LOOP_DURATION-1; j++){
            // finding true onset
            if ((j == 0 || (onsets[i][j-1] < thresh_min || onsets[i][j-1] > thresh_max)) &&
                (onsets[i][j] >= thresh_min && onsets[i][j] <= thresh_max)){ 
                true_onsets[i][j] = 1;
                
                // find the end of the note
                let duration_count = 0;
                for (var k=0; j + k < LOOP_DURATION; k++){
                    if (onsets[i][j+k] >= thresh_min && onsets[i][j+k] <= thresh_max){ 
                        duration_count += 1;
                    } else break;
                }
                true_durs[i][j] = duration_count; // # of 16th note
            }
        } 
      }

      // live.step has mono-phonic sequences (up to 16 tracks)
      for (var k=0; k< 16; k++){ // 16 = number of monophonic sequence in live.step
        var pitch_sequence = [];
        var velocity_sequence = [];
        var duration_sequence = [];
        for (var j=0; j < LOOP_DURATION; j++){

            var count = 0;
            for (var i=0; i< NUM_MIDI_CLASSES; i++){
                if (true_onsets[i][j]) count++; // if there is an onset

                // (count > k) means you need to add another monophonic track 
                // (count <= k) means you have already handled the note 
                if (count > k) {
                    pitch_sequence.push(i + MIN_MIDI_NOTE);
                    velocity_sequence.push(Math.floor(velocities[i][j]*127.));
                    let duration = true_durs[i][j] / 16.0; // 1.0 = whole note
                    duration_sequence.push(Math.min(Math.floor(duration * 127.), 127));
                    break;
                }
            }
            if (count <= k){ // padding if there is no note
                pitch_sequence.push(0);
                velocity_sequence.push(0);
                duration_sequence.push(0);
            }
        }

        // output for live.step object
        Max.outlet("pitch_output", k+1, pitch_sequence.join(" "));
        Max.outlet("velocity_output", k+1, velocity_sequence.join(" "));
        Max.outlet("duration_output", k+1, duration_sequence.join(" "));
    }

      Max.outlet("generated", 1);
      utils.log_status("");
      isGenerating = false;
  } else {
      utils.error_status("Model is not trained yet");
  }
}

// Clear training data 
Max.addHandler("clear_train", ()=>{
    train_data_onsets = [];  // clear
    train_data_velocities = [];
    train_data_timeshift = [];  
    reportNumberOfBars();
});

Max.addHandler("stop", ()=>{
    vae.stopTraining();
});

Max.addHandler("savemodel", (path)=>{
    // check if already trained or not
    if (vae.isReadyToGenerate()){
        filepath = "file://" + path;
        vae.saveModel(filepath);
        utils.log_status("Model saved.");
    } else {
        utils.error_status("Train a model first!");
    }
});

Max.addHandler("loadmodel", (path)=>{
    filepath = "file://" + path;
    vae.loadModel(filepath);
    utils.log_status("Model loaded!");
});

Max.addHandler("epochs", (e)=>{
    vae.setEpochs(e);
    utils.post("number of epochs: " + e);
});

function reportNumberOfBars(){
    Max.outlet("train_bars", train_data_onsets.length * 2);  // number of bars for training
}