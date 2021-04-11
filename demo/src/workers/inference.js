import * as tf from '@tensorflow/tfjs';
let session;

export function loadModel(modelFile) {
    return tf.loadGraphModel(modelFile).then(sess => {
        console.log('Model successfully loaded.')
        session = sess;
        return { result: true };
    }, res => {
        return { result: false, res};
    });
}

export function predict() {
    const tensor = tf.truncatedNormal([1, 64])
    const outputData = session.predict(tensor);
    const rescaled = outputData;
    const resized = tf.image.resizeBilinear(rescaled, [256, 256]);
    const output = resized.gather(0);
    return output.array();
}
