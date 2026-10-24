"""Exceções de domínio do reconhecedor facial.

Cada erro esperado do pipeline tem um tipo próprio, para que a interface consiga
mostrar uma mensagem clara ao usuário sem depender de texto de exceção genérica.
"""


class FaceRecError(Exception):
    """Base de todos os erros de domínio."""

    code = "face_rec_error"
    message = "Erro no reconhecimento facial."

    def __init__(self, message: str | None = None):
        super().__init__(message or self.message)


class NoFaceDetected(FaceRecError):
    code = "no_face"
    message = "Nenhuma face detectada na imagem."


class MultipleFacesDetected(FaceRecError):
    code = "multiple_faces"
    message = "Mais de uma face detectada. Envie uma imagem com apenas uma pessoa."


class LowQualityImage(FaceRecError):
    code = "low_quality"
    message = "Imagem inadequada (face pequena, desfocada ou detecção pouco confiável)."


class InvalidImage(FaceRecError):
    code = "invalid_image"
    message = "Arquivo não é uma imagem válida."


class DuplicatePerson(FaceRecError):
    code = "duplicate_person"
    message = "Esta face é muito parecida com uma pessoa já cadastrada."


class InconsistentEnrollment(FaceRecError):
    code = "inconsistent_enrollment"
    message = "As imagens enviadas no cadastro não parecem ser da mesma pessoa."


class PersonNotFound(FaceRecError):
    code = "person_not_found"
    message = "Pessoa não encontrada."
