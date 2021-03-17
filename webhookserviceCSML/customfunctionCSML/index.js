const https = require('https');

async function getClosestString(event) {
    const {article_id, is_url} = event;
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
}

module.exports.handler = async function handler(event) {
    // We wrap the results in an object stating if the everything went as expected
    if (!event.article_id || !event.is_url) {
        return {
            success: false,
            message: 'You must provide `article_id` and `is_url` parameters',
        };
    }
    return {
        success: true,
        data: await getClosestString(event),
    };
};