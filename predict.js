
const fs = require('fs')
const tf = require('@tensorflow/tfjs-node')

const imageVecs = tf.tensor(JSON.parse(fs.readFileSync('vec.json')))
const metadata = JSON.parse(fs.readFileSync('meta.json'))
let MODEL = null
let MODEL_LOADING = false


async function loadModel() {
    if (!MODEL) {
        const model = await tf.node.loadSavedModel('./glove_word_model')
        MODEL = model
    }
    return MODEL
}

loadModel()


async function search(word, max=100) {
    word = word.toLowerCase()
    const model = await loadModel()
    const input = tf.tensor([[word]])
    const wordVec = await model.predict(input)
    const wordVecArray = await wordVec.array()
    
    let zero = 0
    for (const v of wordVecArray[0]) {
        zero += v * v
    }
    if (zero === 0) {
        return null
    }

    const distances = await tf.sum(wordVec.mul(imageVecs), -1).mul(-1.0).add(1.0).div(2.0).array()
    const distancesWithArg = distances.map((d, i) => [i, d])
    distancesWithArg.sort((a, b) => a[1] - b[1])

    const ret = []
    for (let i = 0; i < max; i++) {
        const [ind, distance] = distancesWithArg[i]
        ret.push({
            ...metadata[ind],
            distance
        })
    }
    return ret
}

module.exports = search
