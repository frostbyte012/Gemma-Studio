
import { useEffect, useState } from 'react';

export type AnimationProps = {
  duration?: number;
  delay?: number;
  direction?: 'up' | 'down' | 'left' | 'right';
  type?: 'fade' | 'scale' | 'slide';
  distance?: number;
};

export const useAnimateOnMount = (options: AnimationProps = {}) => {
  const {
    duration = 300,
    delay = 0,
    direction = 'up',
    type = 'fade',
    distance = 20,
  } = options;
  
  const [isVisible, setIsVisible] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true);
    }, delay);
    
    return () => clearTimeout(timer);
  }, [delay]);
  
  let initialStyles = {};
  let animatedStyles = {};
  
  if (type === 'fade' || type === 'slide') {
    initialStyles = {
      opacity: 0,
      ...initialStyles,
    };
    animatedStyles = {
      opacity: 1,
      ...animatedStyles,
    };
  }
  
  if (type === 'scale') {
    initialStyles = {
      transform: 'scale(0.95)',
      ...initialStyles,
    };
    animatedStyles = {
      transform: 'scale(1)',
      ...animatedStyles,
    };
  }
  
  if (type === 'slide') {
    switch (direction) {
      case 'up':
        initialStyles = {
          transform: `translateY(${distance}px)`,
          ...initialStyles,
        };
        break;
      case 'down':
        initialStyles = {
          transform: `translateY(-${distance}px)`,
          ...initialStyles,
        };
        break;
      case 'left':
        initialStyles = {
          transform: `translateX(${distance}px)`,
          ...initialStyles,
        };
        break;
      case 'right':
        initialStyles = {
          transform: `translateX(-${distance}px)`,
          ...initialStyles,
        };
        break;
    }
    
    animatedStyles = {
      transform: 'translate(0, 0)',
      ...animatedStyles,
    };
  }
  
  const styles = {
    transition: `all ${duration}ms ease-out`,
    ...(isVisible ? animatedStyles : initialStyles),
  };
  
  return { styles, isVisible };
};

export const staggeredChildren = (delay: number = 100) => {
  return (index: number) => ({
    delay: index * delay,
  });
};
