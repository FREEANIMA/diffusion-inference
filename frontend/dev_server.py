import http.server, os

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        clean = self.path.split('?')[0]
        path = self.translate_path(clean)
        has_ext = os.path.splitext(clean)[1] != ''

        # 확장자 없는 경로(/txt2img 같은 페이지 주소)만 index.html로
        if not os.path.exists(path) and not has_ext:
            self.path = '/index.html'
        return super().do_GET()

http.server.test(HandlerClass=SPAHandler, port=5500)