from flask import flash
from flask_wtf import FlaskForm
from wtforms import FileField, StringField, SubmitField
from wtforms.validators import Length, ValidationError


def validate_upload(form, field):
    if not form.text.data and not form.file.data:
        flash('Form must include either text or file.', 'warning')
        raise ValidationError()


class SearchForm(FlaskForm):
    text = StringField('Text', [Length(min=0, max=128)])
    file = FileField('File', [validate_upload])

    search = SubmitField('Search', validators=[])

    def validate(self, extra_validators=None) -> bool:
        """Overriding this definition to validate fields in a specific order, and to halt on a validation error."""
        item_order = [
            'text',
            'file',
        ]
        for item in item_order:
            field = self._fields[item]
            if not field.validate(self, tuple()):
                return False
        return True