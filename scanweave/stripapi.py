"""长幅条带拼接工作台的 HTTP API。"""
import io
from flask import Blueprint, request, jsonify, send_file, Response
from . import validation as v


def create_strip_api(service, assets):
    api = Blueprint('strips', __name__, url_prefix='/api/strip-sheets')

    def body():
        data = request.get_json(silent=True)
        if data is None:
            raise v.Invalid('请提交JSON对象')
        return v.object_value(data)

    def png_response(raw):
        return Response(raw, mimetype='image/png', headers={'Cache-Control': 'no-store'})

    def download(raw, filename, mimetype):
        return send_file(io.BytesIO(raw), as_attachment=True,
                         download_name=filename, mimetype=mimetype)

    @api.get('')
    def list_sheets():
        return jsonify(service.list_sheets(request.args.get('query', '')))

    @api.post('')
    def create_sheet():
        result, status = service.create(body())
        return jsonify(result), status

    @api.get('/<sheet_id>')
    def get_sheet(sheet_id):
        return jsonify(service.get(sheet_id))

    @api.patch('/<sheet_id>')
    def update_sheet(sheet_id):
        return jsonify(service.update(sheet_id, body()))

    @api.post('/<sheet_id>/duplicate')
    def duplicate_sheet(sheet_id):
        result, status = service.duplicate(sheet_id)
        return jsonify(result), status

    @api.delete('/<sheet_id>')
    def delete_sheet(sheet_id):
        return jsonify(service.delete(sheet_id))

    @api.post('/<sheet_id>/undo')
    def undo_sheet(sheet_id):
        return jsonify(service.undo(sheet_id, body(), False))

    @api.post('/<sheet_id>/redo')
    def redo_sheet(sheet_id):
        return jsonify(service.undo(sheet_id, body(), True))

    @api.get('/<sheet_id>/history')
    def sheet_history(sheet_id):
        return jsonify(service.store.history(sheet_id))

    # ---------- 条带 ----------
    @api.post('/<sheet_id>/strips')
    def add_strips(sheet_id):
        result, status = service.add_strips(sheet_id, body())
        return jsonify(result), status

    @api.post('/<sheet_id>/strips/remove')
    def remove_strips(sheet_id):
        return jsonify(service.remove_strips(sheet_id, body()))

    @api.post('/<sheet_id>/strips/reorder')
    def reorder_strips(sheet_id):
        return jsonify(service.reorder(sheet_id, body()))

    @api.patch('/<sheet_id>/strips/<strip_id>')
    def update_strip(sheet_id, strip_id):
        return jsonify(service.update_strip(sheet_id, strip_id, body()))

    @api.get('/<sheet_id>/strips/<strip_id>/thumbnail')
    def strip_thumbnail(sheet_id, strip_id):
        return png_response(service.strip_thumbnail(sheet_id, strip_id))

    # ---------- 分析与裁决 ----------
    @api.post('/<sheet_id>/analyze')
    def analyze(sheet_id):
        return jsonify(service.analyze(sheet_id, body()))

    @api.post('/<sheet_id>/reset')
    def reset_layout(sheet_id):
        return jsonify(service.reset_layout(sheet_id, body()))

    @api.post('/<sheet_id>/edges')
    def set_edge(sheet_id):
        return jsonify(service.set_edge(sheet_id, body()))

    @api.post('/<sheet_id>/issues/<issue_id>/resolve')
    def resolve_issue(sheet_id, issue_id):
        payload = body()
        payload['issue_id'] = issue_id
        return jsonify(service.resolve_issue(sheet_id, payload))

    @api.post('/<sheet_id>/confirm')
    def confirm(sheet_id):
        return jsonify(service.confirm(sheet_id, body()))

    # ---------- 可视化与导出 ----------
    @api.get('/<sheet_id>/preview')
    def preview(sheet_id):
        return png_response(service.preview_png(sheet_id))

    @api.post('/<sheet_id>/export')
    def export_sheet(sheet_id):
        raw, filename, mimetype = service.export(sheet_id, body())
        return download(raw, filename, mimetype)

    @api.get('/<sheet_id>/manifest')
    def manifest(sheet_id):
        return jsonify(service.layout_manifest(sheet_id))

    return api
