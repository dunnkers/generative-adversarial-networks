import React from 'react';
import { Typography } from 'antd';
import './App.css';
import ModelShowcase from './components/ModelShowcase';
import GenerativeShowcase from './components/GenerativeShowcase';
import InferenceWorker from 'workerize-loader!./workers/inference' // eslint-disable-line import/no-webpack-loader-syntax
const { Text, Link, Paragraph } = Typography;


function App() {
  const p = process.env.PUBLIC_URL;
  const worker = InferenceWorker();

  return (
    <article className="App">
      <header className="App-header">
        <h1>Painting Van Goghs using GANs</h1>
        <h4>
          <Text type="secondary">Using Deep Learning with TensorFlow/Keras</Text>
        </h4>
      </header>
      <Paragraph>
        This web page allows you to do live inferences on our trained DCGAN model. The model was trained on a Van Gogh paintings <Link href='https://www.kaggle.com/ipythonx/van-gogh-paintings'>dataset</Link>. The training is still in an early phase, so don't expect real painting-like results. Nonetheless, this page exists to demonstrate the capability of loading our model in TensorFlow.js and doing live inferences. So, enjoy making some art in your browser ✨.
      </Paragraph>
      <ModelShowcase worker={worker} modelFile={p+'/dcgan-gogh/model.json'} >
        <GenerativeShowcase />
      </ModelShowcase>
      <Paragraph type='secondary'>
        Note generating an image might take a couple seconds: some time is required to let the network pass a random latent input vector through all its layers.
      </Paragraph>
      <Paragraph>
        <small>
          <Text type='secondary'>
            &gt; All our code is available on&nbsp;
            <Link href='https://github.com/dunnkers/generative-adversarial-networks/'>Github <img src={p+'/github32.png'} 
              alt='Github logo'
              style={{width: 16, verticalAlign: 'text-bottom'}} />
            </Link>
          </Text>
        </small>
      </Paragraph>
    </article>
  );
}

export default App;
