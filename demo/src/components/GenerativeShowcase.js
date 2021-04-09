import { Button, Empty, Result } from 'antd';
import React, { useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

function GenerativeShowcase(props) {
  const canvasElement = useRef(null);
  const [init, setInit] = useState(false);
  const [loading, setLoading] = useState(false);

  const gen = () => {
    setInit(true);
    setLoading(true);
    const session = props.session;
    const tensor = tf.truncatedNormal([1, 64])
    
    const outputData = session.predict(tensor);
    // const rescaled = outputData.mul(0.5).add(0.5);
    const rescaled = outputData;
    const resized = tf.image.resizeBilinear(rescaled, [256, 256]);
    const output = resized.gather(0);
    tf.browser.toPixels(output, canvasElement.current);
    setLoading(false);
  };
  
  const canv = init ? 'inline' : 'none';
  const empt = init ? 'none' : 'inline';

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
        <Button onClick={gen} disabled={!props.session} loading={loading}>
          Generate image
        </Button>
      }
    />
  );
}

export default GenerativeShowcase;