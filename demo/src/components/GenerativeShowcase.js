import { Button, Empty, Result } from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

function GenerativeShowcase(props) {
  const canvasElement = useRef(null);
  const [state, setState] = useState({
    init: false,
    generating: false,
    sentOffscreen: false
  });
  
  useEffect(() => {
    const hasOffscreen = 'OffscreenCanvas' in window;
    if (!state.sentOffscreen && hasOffscreen) {
      const offscreen = canvasElement.current.transferControlToOffscreen();
      props.worker.postMessage({canvas: offscreen}, [offscreen]);
      setState(state => ({ ...state, sentOffscreen: true }));
    }

    if (!state.generating) return;

    async function infer() {
      await props.worker.predict().then(output => {
        if (output) {
          console.log('Rendering in browser...');
          const tensor = new tf.tensor3d(output);
          tf.browser.toPixels(tensor, canvasElement.current);
        }
        setState({ init: true, generating: false, sentOffscreen: true });
      });
    }
    infer();
  }, [props.worker, state.init, state.generating, state.sentOffscreen]);
  
  const canv = state.init ? 'inline' : 'none';
  const empt = state.init ? 'none' : 'inline';

  return (
    <Result
      icon={<div style={{width: 256, height: 256}}>
        <canvas ref={canvasElement} style={{'display': canv}} />
        <Empty style={{'display': empt}} 
          description={
            <span>
              No image generated yet. Click the button below!
            </span>
          }/>
      </div>}
      extra={
        <Button onClick={() => setState({ ...state, 'generating': true })}
          disabled={!props.session} loading={state.generating}>
          Generate image
        </Button>
      }
    />
  );
}

export default GenerativeShowcase;