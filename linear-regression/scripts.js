import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { mod } from '@tensorflow/tfjs'

window.onload = async () => {
  const xs = [1, 2, 3, 4]
  const ys = [1, 5, 1, 5]

  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xAxisDomain: [0, 10], yAxisDomain: [0, 20] }
  )

  const model = tf.sequential()
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.sgd(0.1)
  })

  const inputs = tf.tensor(xs)
  const labels = tf.tensor(ys)
  await model.fit(inputs, labels, {
    batchSize: 4,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks({ name: '训练过程' }, ['loss'])
  })

  const output = model.predict(tf.tensor([5]))
  output.print()

  xs.push(5)
  ys.push(output.dataSync()[0])
  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: xs.map((x, i) => ({ x, y: ys[i] })) },
    { xAxisDomain: [0, 10], yAxisDomain: [0, 20] }
  )
  alert(`如果输入的是5，返回的是${output.dataSync()[0]}`)
}
