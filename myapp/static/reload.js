if ('performance' in window) {
    const navigationLastEntry = performance.getEntriesByType('navigation').pop();
  
    if (navigationLastEntry
        && navigationLastEntry.type === 'back_forward'
        && navigationLastEntry.unloadEventStart === 0) {
      // Reload the page with a cache-busting query parameter
      window.location.href = `${window.location.href.split('?')[0]}?noCache=${new Date().getTime()}`;
    }
}

