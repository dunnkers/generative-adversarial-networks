import React, { useState, useEffect } from 'react';
import { Alert, Spin } from 'antd';
import GenerativeShowcase from './GenerativeShowcase';

function ModelShowcase(props) {
  const [state, setState] = useState({
    msg: 'Loading model...', loading: true, success: false, session: null,
    feedback: 'Load the model to start making inferences.'
  });

  // Load model
  useEffect(() => {
    if (!state.loading) return; // was not initiated

    props.worker.loadModel(props.modelFile).then(response => {
      const { result } = response;
      if (result) {
        // wait a bit before showing result
        setTimeout(() => {
          setState({
            msg: 'Successfully loaded TensorFlow.js model',
            feedback: 'TensorFlow.js is ready for live inferences.',
            success: true
          });
        }, 1500);
      } else {
        setState({
          msg: 'Oops, model could not be loaded',
          feedback: response.res.message,
          loading: false,
          failure: true
        });
        console.warn('Model failed to load', response.res)
      }
    });
  }, [props.modelFile, state.loading, props.worker]);

  return (
    <div style={{
        background: 'white', margin: '50px 0',
        display: 'flex', flexDirection: 'column', alignItems: 'center'
      }}>
      <div style={{textAlign: 'center'}}>
        <Alert
          message={state.msg}
          // description={<code>{state.feedback}</code>}
          type={state.success ? 'success' : 
            (state.failure ? 'error' : 'info')}
          icon={state.loading ? <Spin style={{ float: 'left' }}/> : undefined}
          showIcon
          style={{ maxWidth: '450px', minWidth: '200px' }}
        />
      </div>

      {props.children && 
      (props.children.map ? props.children : [props.children])
        .map((child, i) => {
        if (child.type === GenerativeShowcase)
          return React.cloneElement(child, {
            key: i,
            session: state.success,
            worker: props.worker
          });
        return child;
      })}
    </div>
  );
}

export default ModelShowcase;