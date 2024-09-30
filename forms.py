from flask import flash
from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    FileField,
    IntegerField,
    StringField,
    SubmitField
)
from wtforms.validators import Length, NumberRange, Optional, ValidationError

from configs import CONSTS
from consts import clip_valid_extensions


def validate_upload(form, field):
    data = []
    for field in CONSTS.form_fields:
        data.append(form.__getattribute__(field).data)

    if not any(data):
        flash('Form must include search parameter.', 'warning')
        raise ValidationError()

    if (CONSTS.clip or CONSTS.hash):
        if form.file.data and not form.file.data.filename.endswith(clip_valid_extensions):
            flash('Invalid image type.', 'warning')
            raise ValidationError()

        if any([form.search_average_hash.data, form.search_colorhash.data, form.search_crop_resistant_hash.data, form.clip_file.data]) and not form.file.data:
            flash('A file is required for the selected search type.', 'warning')
            raise ValidationError()


class SearchForm(FlaskForm):
    file = FileField('File (Drag \'n Drop)', [validate_upload])
    search_average_hash = BooleanField('Average Hash', default=True)
    search_colorhash = BooleanField('Color Hash', default=False)
    search_crop_resistant_hash = BooleanField('Crop Resistant Hash', default=False)
    clip_file = BooleanField('CLIP File', default=False)
    clip_text = StringField('CLIP Text', [Length(min=0, max=128)])
    exif_text = StringField('Exif Values', [Length(min=0, max=128)], render_kw={'placeholder': 'User Comment, Image Description'})
    ocr_text = StringField('OCR Text', [Length(min=0, max=128)])
    min_face_count = IntegerField('Face Count, Min', [Optional(), NumberRange(min=0, max=20)])
    max_face_count = IntegerField('Face Count, Max', [Optional(), NumberRange(min=0, max=20)])

    search = SubmitField('Search', validators=[])

    def validate(self, extra_validators=None) -> bool:
        """Overriding this definition to validate fields in a specific order, and to halt on a validation error."""
        item_order = [
            'file',
            'search_average_hash',
            'search_colorhash',
            'search_crop_resistant_hash',
            'clip_file',
            'clip_text',
            'exif_text',
            'ocr_text',
            'min_face_count',
            'max_face_count',
        ]
        for item in item_order:
            field = self._fields[item]
            if not field.validate(self, tuple()):
                return False
        return True
