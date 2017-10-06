# deeplearning

- classifier  : 各種分類器を保存｡
- graph : 各種グラフ関数を定義。

---
- 各種ライブラリ（scikit-learn, tensorflow, kerasなど）はファイル名最後に省略形で表記。

***_sl = scikit-learn

***_ts = tensorflow

***_ke = keras

など。

---

# Tensorflow

### 行列の生成

    tf.diag([0.5,0.5,0.5])  #体格行列
    tf.truncated_normal([2,3])
    tf.fill([2,3],5.0)
    tf.random_uniform([3,2])
    tf.convert_to_tensor(np.array(....))
    
### 行列の操作
    tf.matmul(A,B)
    tf.transpose(A)
    tf.matrix_determinant(A)
    tf.matrix_inverse(A)
    tf.cholesky(A)
    eigenvalues, engenvectors = tf.self_adjoint_eig(D)
    
## 活性化関数の実装
    import tensorflow.nn as nn
    tf.nn.relu([...])   # min(0, x)
    tf.nn.relul6([...]) # min(max(0,x), 6)
    tf.nn.sigmoid([...])
    tf.nn.tanh([...])
    tf.nn.softsign([...])
    tf.nn.softplus([...])
    tf.nn.elu([...])
    
    
    