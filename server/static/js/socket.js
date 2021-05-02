$(document).ready(function () { // 等待页面全部加载完再执行函数
    console.log('Ready!');

    namespace = '/message';
    console.log(location.protocol + '//' + document.domain + ':' + location.port + namespace);
    var socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port + namespace);
    socket.on('server_response', function(res) {
        // console.log(res.data);
        $('#kws_re').text(res.data);
    });
});