const https = require('https');

const getSimilarArticle = () => {

    return new Promise((resolve, reject) => {

        const data = JSON.stringify({
            number: '8',
        })

        var options = {
            hostname: '521a02e992b8.ngrok.io',
            port: 443,
            path: '/webhook',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            }
        };

        var req = https.request(options, (res) => {
            console.log('statusCode:', res.statusCode);
            console.log('headers:', res.headers);

            res.on('data', (d) => {
                process.stdout.write(d);
            });
        });
        req.on('error', (e) => {
            console.error(e);
        });
        req.write(data);
        req.end();

    });

};

module.exports.handler = (event) => {
    return getSimilarArticle();
}