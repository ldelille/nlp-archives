const {handler} = require('./testcsml2');

(async () => {
    const user = await handler();
    console.log(user);
})();