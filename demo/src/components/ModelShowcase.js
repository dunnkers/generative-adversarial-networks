import React, { useState, useEffect } from 'react';
import { Alert, Spin } from 'antd';
import GenerativeShowcase from './GenerativeShowcase';
import * as tf from '@tensorflow/tfjs';

function ModelShowcase(props) {
  const [state, setState] = useState({
    msg: 'Loading model...', loading: true, success: false, session: null,
    feedback: 'Load the model to start making inferences.'
  });

  // Load model
  useEffect(() => {
    if (!state.loading) return; // was not initiated

    tf.loadGraphModel(props.modelFile).then(session => {
      console.log('Model successfully loaded.')

      // wait a bit before showing result
      setTimeout(() => {
        setState({
          msg: 'Successfully loaded TensorFlow.js model',
          feedback: 'TensorFlow.js is ready for live inferences.',
          // loading: false,
          success: true,
          session
        });
      }, 1500);
    }, res => {
      setState({
        msg: 'Oops, model could not be loaded',
        feedback: res.message,
        loading: false,
        failure: true
      });
      console.warn('Model failed to load', res)
    });
  }, [props.modelFile, state.loading]);

  return (
    <div style={{ background: 'white', margin: '50px 0' }}>
      <div style={{textAlign: 'center'}}>
        <Alert
          message={state.msg}
          // description={<code>{state.feedback}</code>}
          type={state.success ? 'success' : 
            (state.failure ? 'error' : 'info')}
          icon={state.loading ? <Spin style={{ float: 'left' }}/> : undefined}
          showIcon
        />
      </div>

      {props.children && 
      (props.children.map ? props.children : [props.children])
        .map((child, i) => {
        if (child.type === GenerativeShowcase)
          return React.cloneElement(child, {
            key: i,
            session: state.session,
            model: props.model,
            crop: props.crop
          });
        return child;
      })}
    </div>
  );
}

export default ModelShowcase;