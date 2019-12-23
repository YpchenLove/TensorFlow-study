import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { mod } from '@tensorflow/tfjs'

window.onload = async () => {
  const heights = [150, 160, 170]
  const weights = [40, 50, 60]

  tfvis.render.scatterplot(
    { name: '身高体重训练集' },
    { values: heights.map((x, i) => ({ x, y: weights[i] })) },
    { xAxisDomain: [140, 180], yAxisDomain: [30, 70] }
  )

  const inputs = tf
    .tensor(heights)
    .sub(150)
    .div(20)
  const labels = tf
    .tensor(weights)
    .sub(40)
    .div(20)
  inputs.print()
  labels.print()

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  })

  await model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss'])
  })

  const output = model.predict(
    tf
      .tensor([180])
      .sub(150)
      .div(20)
  )
  output.print()

  heights.push(180)
  heights.push(output.dataSync()[0])
  tfvis.render.scatterplot(
    { name: '身高体重训练集' },
    { values: heights.map((x, i) => ({ x, y: heights[i] })) },
    { xAxisDomain: [140, 200], yAxisDomain: [30, 70] }
  )
  alert(
    `如果输入的是180，返回的是${
      output
        .mul(20)
        .add(40)
        .dataSync()[0]
    }`
  )
}
