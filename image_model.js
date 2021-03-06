
const tf = require('@tensorflow/tfjs-node')

// input_image = tf.keras.layers.Input((32, 32, 1), dtype=tf.float32)
// x = input_image
// x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
// x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
// x = tf.keras.layers.BatchNormalization()(x)
// x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
// x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
// x = tf.keras.layers.BatchNormalization()(x)
// x = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(x)
// x = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(x)
// x = tf.keras.layers.BatchNormalization()(x)
// x = tf.keras.layers.Flatten()(x)
// x = tf.keras.layers.Dense(256, activation='relu')(x)
// x = tf.keras.layers.Dense(100, activation='linear')(x)
// x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, -1))(x)
// image_model = tf.keras.Model(inputs=input_image, outputs=x)
// assert image_model(tf.random.uniform((1, 32, 32, 1))).shape == (1, 100)

class L2Normalize extends tf.layers.Layer {
    constructor (config) {
        super(config)
    }
  
    call (input) {
        return tf.tidy(() => {
            let x = input[0]
            let k = tf.square(x)
            k = tf.sum(k, -1, true)
            k = tf.sqrt(k)
            return x.div(k)
        })
    }

    static get className () {
        return 'L2Normalize';
    }
}

tf.serialization.registerClass(L2Normalize)

const model = tf.sequential()
model.add(tf.layers.inputLayer({ inputShape: [32, 32, 1] }))
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }))
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }))
model.add(tf.layers.batchNormalization())
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }))
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu', padding: 'same' }))
model.add(tf.layers.batchNormalization())
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 2, activation: 'relu', padding: 'same' }))
model.add(tf.layers.conv2d({ filters: 32, kernelSize: 2, activation: 'relu', padding: 'same' }))
model.add(tf.layers.batchNormalization())
model.add(tf.layers.flatten())
model.add(tf.layers.dense({ units: 256, activation: 'relu' }))
model.add(tf.layers.dense({ units: 100, activation: 'linear' }))
model.add(new L2Normalize())

const inputs = tf.randomUniform([2, 32, 32, 1])
const ret = model.predict(inputs)
console.log(ret.shape)
ret.print()
