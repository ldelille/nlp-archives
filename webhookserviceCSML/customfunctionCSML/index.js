const https = require('https');


async function getClosestString(event) {
    const {article_id, is_url} = event;
    let response = '';
    return new Promise((resolve, reject) => {
        var postData = JSON.stringify({
            'article_id': article_id,
            'is_url': is_url
        });
        var options = {
            hostname: '58caef3a0c0a.ngrok.io',
            port: 443,
            path: '/webhook',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': postData.length
            }
        };
        var req = https.request(options, (res) => {
            res.on('data', (d) => {
                response += d.toString();
            });
            res.on('close', () => {
                console.log(response)
                resolve(JSON.parse(response));
            })
        });
        req.on('error', (e) => {
            console.error(e);
        });
        req.write(postData);
        req.end();
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