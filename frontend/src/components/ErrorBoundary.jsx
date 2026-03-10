import React from 'react';
import { childLogger, serializeError } from '../logger';

const boundaryLogger = childLogger({ area: 'react_error_boundary' });

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    boundaryLogger.error(
      {
        err: serializeError(error),
        componentStack: info?.componentStack || '',
      },
      'react_render_error'
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="page">
          <div className="error-banner">
            The app hit an unexpected error. Refresh and try again. If it keeps happening, check the browser console
            logs for details.
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
