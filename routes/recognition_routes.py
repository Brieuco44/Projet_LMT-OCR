from flask import Blueprint, request, jsonify, session

recognition_bp = Blueprint("recognition", __name__)

@recognition_bp.route('/test', methods=['POST'])
def test():
    pass