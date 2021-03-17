const https = require('https');

const getRandomUser = () => {

  let response = '';
  return new Promise((resolve, reject) => {
    const req = https
      .get('https://randomuser.me/api', (res) => {
        res.on('data', (d) => {
          response += d.toString();
        });
        res.on('close', () => {
          resolve(JSON.parse(response).results[0]);
        })
      })
      .on('error', (e) => {
        reject(e);
      });
  });

};

module.exports.handler = (event) => {
  return getRandomUser();
}