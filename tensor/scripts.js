import * as tf from '@tensorflow/tfjs'

const t0 = tf.tensor(1)

t0.print()

console.log(t0)

const t1 = tf.tensor([1, 2, 3])

t1.print()

console.log(t1)

const t2 = tf.tensor([
  [1, 2, 34],
  [3, 4, 2]
])

t2.print()

console.log(t2)

const t3 = tf.tensor([
  [
    [1, 2],
    [3, 4]
  ],
  [
    [5, 6],
    [7, 8]
  ]
])

t3.print()

console.log(t3)
