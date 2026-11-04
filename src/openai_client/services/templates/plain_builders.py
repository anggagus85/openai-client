"""Plain-text email builders."""

from openai_client.configs import settings
from openai_client.services.templates.styles import _TemplateStyles


class _PlainTemplateMixin(_TemplateStyles):
    """Plain-text email builder methods."""

    def build_plain_footer(self, year: int) -> str:
        """
        Build the shared plain-text footer block.

        Args:
            year: The current year for copyright.

        Returns:
            str: The plain-text footer, terminated by a newline.
        """
        return (
            f"---\n© {year} BaliRoam. All rights reserved.\n"
            "Your portal to Bali's finest travel experiences.\n"
        )

    def build_verification_email_plain(
        self,
        greet_name: str,
        verification_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the verification email.

        Args:
            greet_name: The user's greeting name.
            verification_link: The email verification link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            f"Welcome to BaliRoam, {greet_name}!\n\n"
            "Thank you for embarking on your journey with us. To ensure the security "
            "of your account and access all our travel features, please verify your "
            "email address by visiting the link below:\n\n"
            f"{verification_link}\n\n"
            f"Note: This link will expire in {self._expiry_phrase(expiry_hours)}. "
            "If you didn't create an account, you can safely ignore this email.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_password_reset_email_plain(
        self,
        greet_name: str,
        reset_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the password reset email.

        Args:
            greet_name: The user's greeting name.
            reset_link: The password reset link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            "Reset Your Password\n\n"
            f"Hi {greet_name},\n\n"
            "We received a request to reset your BaliRoam account password. "
            "Use the link below to create a new password:\n\n"
            f"{reset_link}\n\n"
            f"This link will expire in {self._expiry_phrase(expiry_hours)}. "
            "If you didn't request this reset, you can safely ignore this email.\n\n"
            "[SECURITY TIP]\n"
            "Never share this link with anyone. BaliRoam will never ask for your password "
            "via email.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_password_change_email_plain(
        self,
        greet_name: str,
        change_time: str,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the password change confirmation email.

        Args:
            greet_name: The user's greeting name.
            change_time: The formatted change timestamp.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            "Password Changed Successfully\n\n"
            f"Hi {greet_name},\n\n"
            "Your BaliRoam account password was changed successfully. "
            "You will need to sign in again on all other devices with your new password.\n\n"
            f"Change Time: {change_time}\n\n"
            "[IMPORTANT NOTICE]\n"
            "If you didn't make this change, please contact us immediately at "
            f"{settings.COMPANY_TARGET_EMAIL} to secure your account.\n\n"
            f"{self.build_plain_footer(year)}"
        )

    def build_review_confirmation_email_plain(
        self,
        guest_name: str,
        year: int,
    ) -> str:
        """
        Build the plain-text alternative of the review confirmation email.

        Args:
            guest_name: The guest's display name.
            year: The current year for copyright.

        Returns:
            str: The complete text/plain email body.
        """
        return (
            f"Thank You, {guest_name}!\n\n"
            "We've received your review submission and truly appreciate you "
            "taking the time to share your experience with us.\n\n"
            "Our team will review your feedback shortly. Your insights help us "
            "improve and assist other travelers in planning their perfect Bali adventure.\n\n"
            "[NOTE]\n"
            "If you did not recently submit a review on BaliRoam, please disregard "
            "this email — no action is required on your part.\n\n"
            f"{self.build_plain_footer(year)}"
        )
