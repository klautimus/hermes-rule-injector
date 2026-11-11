/**
 * Rule-Injector plugin entry — renders the standalone rule-injector.html in an iframe.
 * The actual rule-injector UI is a self-contained HTML page at /rule-injector.
 */
(function () {
  var SDK = window.__HERMES_PLUGIN_SDK__;
  var React = SDK.React;
  var h = React.createElement;
  var useEffect = React.useEffect;
  var useRef = React.useRef;

  function RuleInjectorIframe() {
    var containerRef = useRef(null);

    useEffect(function () {
      function resize() {
        var el = containerRef.current;
        if (!el) return;
        var parent = el.parentElement;
        while (parent && parent.clientHeight === 0) {
          parent = parent.parentElement;
        }
        if (parent) {
          var rect = parent.getBoundingClientRect();
          el.style.height = rect.height + 'px';
        }
      }
      resize();
      window.addEventListener('resize', resize);
      var timer = setTimeout(resize, 100);
      return function () {
        window.removeEventListener('resize', resize);
        clearTimeout(timer);
      };
    }, []);

    return h('div', {
      ref: containerRef,
      style: {
        width: '100%',
        height: '100%',
        position: 'relative',
        minHeight: 'calc(100vh - 80px)',
      }
    }, h('iframe', {
      src: '/rule-injector',
      style: {
        position: 'absolute',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        border: 'none',
        display: 'block',
      },
      title: 'Rule Injector Dashboard'
    }));
  }

  window.__HERMES_PLUGINS__.register('rule-injector', RuleInjectorIframe);
})();
