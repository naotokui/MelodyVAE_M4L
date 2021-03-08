

const MIN_MIDI_NOTE = 48;
const MAX_MIDI_NOTE = 71;

const NUM_MIDI_CLASSES = MAX_MIDI_NOTE - MIN_MIDI_NOTE + 1; // 24

const LOOP_DURATION = 32; // 2bars x 16th note

const ORIGINAL_DIM = NUM_MIDI_CLASSES * LOOP_DURATION;

const MIN_ONSETS_THRESHOLD = 5; // ignore loops with onsets less than this num

exports.INSTRUMENTS_MELODY = [ 0,1,2,3,4,5,6,7, // piano
    17,18,19,20,21,22,23, // organ
    33,34,35,36,37,38,39, // bass
    40,41,42,43,44, // strings
    64,65,66,67,68,69,70,71, // pipe
    80,81,82,83,84,85,86,87, // lead
    88,89,90,91,92,93,94,95 // pad
 ];

exports.MIN_MIDI_NOTE = MIN_MIDI_NOTE;
exports.MAX_MIDI_NOTE = MAX_MIDI_NOTE;
exports.NUM_MIDI_CLASSES = NUM_MIDI_CLASSES;

exports.LOOP_DURATION = LOOP_DURATION;
exports.ORIGINAL_DIM = ORIGINAL_DIM;

exports.MIN_ONSETS_THRESHOLD = MIN_ONSETS_THRESHOLD;

