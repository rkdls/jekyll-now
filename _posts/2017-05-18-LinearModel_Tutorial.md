---
layout: post
title: TensorFlow Linear Model Tutorial
description: "TensorFlow Linear Model Tutorial"
tags: [machine-learning]
---
<h1>TensorFlow Linear Model Tutorial </h1>


### 텐서플로우 공식홈페이지에나와있는 튜토리얼인 Linear Model을 구현해보자.



```python
import tempfile
import urllib.request
train_file = 'CensusData/trainFile/data.txt'
test_file ='CensusData/testFile/data.txt'
local_file, headers = urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data")
html = open(local_file)


with open('CensusData/trainFile/data.txt', mode='w+') as f:
    f.write(html.read())

local_file, headers = urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test")
html = open(local_file)
with open('CensusData/testFile/data.txt', mode='w+') as f:
    f.write(html.read())
```


```python
import pandas as pd
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, skiprows=1)
```

<h3>Label 칼럼을 추가해보자. 값이 1이면 income(수익)이 50K이상이고, 아니면 0이다.</h3>


```python
LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
df_test[LABEL_COLUMN] = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
```

<h3>연속된 값을 갖는 컬럼과, 단일된 값을 갖는 컬럼으로 구분해보자</h3>


```python
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
```

<H3>데이터를 Tensors로 변환하자.!</H3>


```python
import tensorflow as tf

def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def train_input_fn():
  return input_fn(df_train)

def eval_input_fn():
  return input_fn(df_test)
```

<h3>컬럼을 구분할때 키값을 0또는 1로 바꾸어서 연산하기 쉽게만들어보자</h3>
<h4>예를 들면, gender 컬럼에 Female과 Male값이 들어올수있다면, Female은 0으로, Male은 1로 할당한다</h4>


```python
gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["Female", "Male"])
```

<h3>밸류값을 모른다면(female,male이런것들) hash map으로 변경해서 계산할수도있다.</h3>


```python
education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size=1000)
```

<h3>각 피쳐값들을 텐서형으로 변환해보자.</h3>


```python
race = tf.contrib.layers.sparse_column_with_hash_bucket("race", hash_bucket_size=100)
marital_status = tf.contrib.layers.sparse_column_with_hash_bucket("marital_status", hash_bucket_size=100)
relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size=100)
workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
native_country = tf.contrib.layers.sparse_column_with_hash_bucket("native_country", hash_bucket_size=1000)
```


```python
age = tf.contrib.layers.real_valued_column("age")
education_num = tf.contrib.layers.real_valued_column("education_num")
capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")
```

<h3>Age를 단일된값(categorical)으로 나누어보자 
<br>나이가 오를수록 소득이 오르는게 완전 정비례는 아니니깐</h3>

```python
age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

<H3>열을 합칠수가있다.</H3>


```python
education_x_occupation = tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4))
```

<h3>두개 이상을 합칠수가있다.</h3>


```python
age_buckets_x_education_x_occupation = tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6))
```

<H3>모델만들고 저장하기</H3>


```python
model_dir = 'CensusData/model'
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  model_dir=model_dir)
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_master': '', '_evaluation_master': '', '_environment': 'local', '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1
    }
    , '_task_id': 0, '_num_worker_replicas': 0, '_is_chief': True, '_model_dir': None, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x000002152BA17A20>, '_tf_random_seed': None, '_save_checkpoints_steps': None, '_task_type': None, '_keep_checkpoint_every_n_hours': 10000, '_num_ps_replicas': 0, '_save_checkpoints_secs': 600, '_save_summary_steps': 100, '_keep_checkpoint_max': 5}
    

<h3>훈련시키기</h3>


```python
m.fit(input_fn=train_input_fn, steps=200)
```




    LinearClassifier(params={'feature_columns': [_SparseColumnKeys(column_name='gender', is_integerized=False, bucket_size=None, lookup_config=_SparseIdLookupConfig(vocabulary_file=None, keys=('Female', 'Male'), num_oov_buckets=0, vocab_size=2, default_value=-1), combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='native_country', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='workclass', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='marital_status', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='race', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _BucketizedColumn(source_column=_RealValuedColumn(column_name='age', dimension=1, default_value=None, dtype=tf.float32, normalizer=None), boundaries=(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)), _CrossedColumn(columns=(_SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string)), hash_bucket_size=10000, hash_key=None, combiner='sum', ckpt_to_load_from=None, tensor_name_in_ckpt=None), _CrossedColumn(columns=(_BucketizedColumn(source_column=_RealValuedColumn(column_name='age', dimension=1, default_value=None, dtype=tf.float32, normalizer=None), boundaries=(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)), _SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string)), hash_bucket_size=1000000, hash_key=None, combiner='sum', ckpt_to_load_from=None, tensor_name_in_ckpt=None)], 'gradient_clip_norm': None, 'head': <tensorflow.contrib.learn.python.learn.estimators.head._BinaryLogisticHead object at 0x000002152BA179B0>, 'joint_weights': False, 'optimizer': None})



<h3>평가하기</h3>


```python
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```
    accuracy: 0.836067
    accuracy/baseline_label_mean: 0.236226
    accuracy/threshold_0.500000_mean: 0.836067
    auc: 0.883805
    global_step: 400
    labels/actual_label_mean: 0.236226
    labels/prediction_mean: 0.239753
    loss: 0.35209
    precision/positive_threshold_0.500000_mean: 0.707291
    recall/positive_threshold_0.500000_mean: 0.522101
    

<h3>Regulaization해보기</h3>

<h4>오버피팅을 피하자.</h4>


```python
model_dir = 'CensusData/model'
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  gender, native_country, education, occupation, workclass, marital_status, race,
  age_buckets, education_x_occupation, age_buckets_x_education_x_occupation],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.1,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)
m.fit(input_fn=train_input_fn, steps=200)
```


    LinearClassifier(params={'joint_weights': False, 'head': <tensorflow.contrib.learn.python.learn.estimators.head._BinaryLogisticHead object at 0x000001F09222A780>, 'optimizer': <tensorflow.python.training.ftrl.FtrlOptimizer object at 0x000001F09222A710>, 'gradient_clip_norm': None, 'feature_columns': [_SparseColumnKeys(column_name='gender', is_integerized=False, bucket_size=None, lookup_config=_SparseIdLookupConfig(vocabulary_file=None, keys=('Female', 'Male'), num_oov_buckets=0, vocab_size=2, default_value=-1), combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='native_country', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='workclass', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='marital_status', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='race', is_integerized=False, bucket_size=100, lookup_config=None, combiner='sum', dtype=tf.string), _BucketizedColumn(source_column=_RealValuedColumn(column_name='age', dimension=1, default_value=None, dtype=tf.float32, normalizer=None), boundaries=(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)), _CrossedColumn(columns=(_SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string)), hash_bucket_size=10000, hash_key=None, combiner='sum', ckpt_to_load_from=None, tensor_name_in_ckpt=None), _CrossedColumn(columns=(_BucketizedColumn(source_column=_RealValuedColumn(column_name='age', dimension=1, default_value=None, dtype=tf.float32, normalizer=None), boundaries=(18, 25, 30, 35, 40, 45, 50, 55, 60, 65)), _SparseColumnHashed(column_name='education', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string), _SparseColumnHashed(column_name='occupation', is_integerized=False, bucket_size=1000, lookup_config=None, combiner='sum', dtype=tf.string)), hash_bucket_size=1000000, hash_key=None, combiner='sum', ckpt_to_load_from=None, tensor_name_in_ckpt=None)]})




```python
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```

    accuracy: 0.833057
    accuracy/baseline_label_mean: 0.236226
    accuracy/threshold_0.500000_mean: 0.833057
    auc: 0.876523
    global_step: 1000
    labels/actual_label_mean: 0.236226
    labels/prediction_mean: 0.241978
    loss: 0.362289
    precision/positive_threshold_0.500000_mean: 0.710134
    recall/positive_threshold_0.500000_mean: 0.49558
    
