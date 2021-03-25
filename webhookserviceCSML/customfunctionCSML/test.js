const { handler } = require('./index');

(async () => {
  const user = await handler({
    article_id: '81',
    is_url: ['Yes', 'No'],
  });
  console.log(user);
})();