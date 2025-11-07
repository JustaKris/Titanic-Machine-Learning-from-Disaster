// MathJax configuration for mathematical formulas
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Add document ready handler
document$.subscribe(() => {
  // Re-render MathJax when navigating
  MathJax.typesetPromise();
});
