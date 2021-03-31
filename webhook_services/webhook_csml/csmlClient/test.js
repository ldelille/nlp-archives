const { handler } = require('./index');

(async () => {
  const user = await handler();
  console.log(user);
})();