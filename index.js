function setup() {
//   const num = tf.tensor2d([1, 2.3, 3, 4.5, 5, 6], [3, 2], "int32");
  const model = tf.sequential();
  const configHidden = {
    units: 3,
    inputShape: [3],
    activation: "sigmoid",
  };
  const configOutput = {
    units: 3,
    activation: "sigmoid",
  };

  const hidden = tf.layers.dense(configHidden);
  const output = tf.layers.dense(configOutput);

  model.add(hidden);
  model.add(output);

  const config = {
    optimizer: tf.train.sgd(0.1),
    loss: "meanSquaredError",
  };

  model.compile(config);


  const xs = tf.tensor2d([
    [0.25, 0.92, 0.58],
    [0.15, 0.72, 0.25],
    [0.48, 0.28, 0.17],
   
  ]);

  const ys = tf.tensor2d([
    [0.36, 0.82, 0.83],
    [0.28, 0.52, 0.82],
    [0.99, 0.98, 0.88],
    
  ]);

  async function train(){
    for(let i=0; i<1000; i++){
        const response = await model.fit(xs, ys);
        console.log(`${i} - ${response.history.loss[0]}`);
    }
    
  }
  async function predicted (){
    await train();
    await console.log("Training Completed!");
    let outputs = model.predict(xs);
    outputs.print();
  }

  predicted();
  

//   let outputs = model.predict(inputs);
//   outputs.print();

}

setup();
