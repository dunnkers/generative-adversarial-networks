import { Button } from 'antd';
import React from 'react';

function GenerativeShowcase(props) {
  const gen = () => {
    console.log(props.session);
  };

  return (
    <div>
      <Button onClick={() => gen()}>
        Generate image
      </Button>
    </div>
  );
}

export default GenerativeShowcase;