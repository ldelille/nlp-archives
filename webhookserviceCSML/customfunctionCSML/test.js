const { handler } = require('./index');

(async () => {
  const user = await handler({
    article_id: 'Yep',
    is_url: ['Yes', 'No'],
  });
  console.log(user);
})();