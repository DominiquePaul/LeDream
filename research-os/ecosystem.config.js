module.exports = {
  apps: [{
    name: 'ml-viz-dev',
    script: 'node_modules/.bin/next',
    args: 'dev --port 3000',
    cwd: '/Users/dpaul/Documents/Dream Machines/05 Research papers/ClaudeRL/method-visualizations',
    autorestart: true,
    max_restarts: 10,
    env: { NODE_ENV: 'development', PORT: '3000' }
  }]
};
