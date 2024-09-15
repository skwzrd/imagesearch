from flask import flash
from flask_wtf import FlaskForm
from wtforms import FileField, StringField, SubmitField
from wtforms.validators import Length, ValidationError

from consts import clip_valid_extensions


def validate_upload(form, field):
    data = [form.clip.data, form.exif.data, form.ocr.data, form.file.data]

    if not any(data):
        flash('Form must include search parameter.', 'warning')
        raise ValidationError()
    
    if len([x for x in data if x]) > 1:
        flash('We only support searching by a single parameter at the moment.', 'warning')
        raise ValidationError()

    if form.file.data and not form.file.data.filename.endswith(clip_valid_extensions):
        flash('Invalid image type.', 'warning')
        raise ValidationError()


class SearchForm(FlaskForm):
    clip = StringField('CLIP', [Length(min=0, max=128)])
    exif = StringField('Exif', [Length(min=0, max=128)])
    ocr = StringField('OCR', [Length(min=0, max=128)])
    file = FileField('File', [validate_upload])

    search = SubmitField('Search', validators=[])

    def validate(self, extra_validators=None) -> bool:
        """Overriding this definition to validate fields in a specific order, and to halt on a validation error."""
        item_order = [
            'clip',
            'exif',
            'ocr',
            'file',
        ]
        for item in item_order:
            field = self._fields[item]
            if not field.validate(self, tuple()):
                return False
        return True