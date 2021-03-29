import React from 'react';
import { Typography } from 'antd';
import './App.css';
import ModelShowcase from './components/ModelShowcase';
import GenerativeShowcase from './components/GenerativeShowcase';
const { Text } = Typography;

function App() {
  const p = process.env.PUBLIC_URL;

  return (
    <article className="App">
      <header className="App-header">
        <h1>GAN's</h1>
        <h4>
          <Text type="secondary">Using Deep Learning with Keras</Text>
        </h4>
      </header>
      <ModelShowcase modelFile={p+'/mnist/model.json'} model={0}>

        <GenerativeShowcase />
      </ModelShowcase>
    </article>
  );
}

export default App;
