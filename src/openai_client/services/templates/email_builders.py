"""Concrete HTML email builders for account and review flows."""

from html import escape

from openai_client.configs import settings
from openai_client.services.templates.html_builders import _HtmlTemplateMixin


class _EmailHtmlMixin(_HtmlTemplateMixin):
    """Concrete HTML email bodies."""

    def build_verification_email(
        self,
        greet_name: str,
        verification_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the complete verification email HTML.

        Args:
            greet_name: The user's greeting name.
            verification_link: The email verification link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)
        button_html = self.build_button(verification_link, "Verify Email Address")

        content_html = self.build_content_wrapper(
            heading=f"Welcome to BaliRoam, {escape(greet_name)}!",
            paragraphs=[
                (
                    "Thank you for embarking on your journey with us. To ensure the security "
                    "of your account and access all our travel features, please verify your "
                    "email address by clicking the button below."
                ),
            ],
            button_html=button_html,
            expiry_note=f"This link will expire in {self._expiry_phrase(expiry_hours)}.<br>"
            "If you didn't create an account, you can safely ignore this email.",
        )

        return self.build_template(
            title="Verify Your BaliRoam Account",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_password_reset_email(
        self,
        greet_name: str,
        reset_link: str,
        expiry_hours: int,
        year: int,
    ) -> str:
        """
        Build the complete password reset email HTML.

        Args:
            greet_name: The user's greeting name.
            reset_link: The password reset link.
            expiry_hours: Hours until the link expires.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)
        button_html = self.build_button(reset_link, "Reset Password")

        security_message = (
            "<strong>Security Tip:</strong><br>"
            "Never share this link with anyone. BaliRoam will never ask for your password via email."
        )
        security_box_html = self.build_security_box(security_message)

        content_html = self.build_content_wrapper(
            heading="Reset Your Password",
            paragraphs=[
                f"Hi {escape(greet_name)},",
                (
                    "We received a request to reset your BaliRoam account password. "
                    "Click the button below to create a new password:"
                ),
            ],
            button_html=button_html,
            expiry_note=f"This link will expire in {self._expiry_phrase(expiry_hours)}.<br>"
            "If you didn't request this reset, you can safely ignore this email.",
            extra_html=security_box_html,
        )

        return self.build_template(
            title="Reset Your BaliRoam Password",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_password_change_email(
        self,
        greet_name: str,
        change_time: str,
        year: int,
    ) -> str:
        """
        Build the complete password change confirmation email HTML.

        Args:
            greet_name: The user's greeting name.
            change_time: The formatted change timestamp.
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)

        info_content = f"<strong>Change Time:</strong> {escape(change_time)}"
        info_box_html = self.build_info_box(info_content)

        escaped_email = escape(settings.COMPANY_TARGET_EMAIL)
        security_message = (
            "<strong>Didn't make this change?</strong><br>"
            "If you didn't change your password, please contact us immediately at "
            f'<a href="mailto:{escaped_email}" style="color: #1a2a6c;">'
            f"{escaped_email}</a> to secure your account."
        )
        security_box_html = self.build_security_box(security_message)

        content_html = self.build_content_wrapper(
            heading="Password Changed Successfully",
            paragraphs=[
                f"Hi {escape(greet_name)},",
                "Your BaliRoam account password was changed successfully.",
                "You will need to sign in again on all other devices with your new password.",
            ],
            extra_html=f"{info_box_html}\n            {security_box_html}",
        )

        return self.build_template(
            title="Your BaliRoam Password Was Changed",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )

    def build_review_confirmation_email(
        self,
        guest_name: str,
        year: int,
    ) -> str:
        """
        Build the review submission confirmation email HTML.

        Sends a branded confirmation to a guest who submitted a review,
        with a disclaimer for recipients who did not initiate the review.

        Args:
            guest_name: The guest's display name (will be escaped).
            year: The current year for copyright.

        Returns:
            str: The complete HTML email.
        """
        header_html = self.build_header()
        footer_html = self.build_footer(year)

        info_message = (
            "<strong>Didn't submit a review?</strong><br>"
            "If you did not recently submit a review on BaliRoam, "
            "please disregard this email &mdash; no action is required on your part. "
            "Your email address may have been entered by mistake."
        )
        info_box_html = self.build_info_box(info_message)

        content_html = self.build_content_wrapper(
            heading=f"Thank You, {escape(guest_name)}!",
            paragraphs=[
                (
                    "We've received your review submission and truly appreciate you "
                    "taking the time to share your experience with us."
                ),
                (
                    "Our team will review your feedback shortly. Your insights help us "
                    "improve and assist other travelers in planning their perfect Bali adventure."
                ),
            ],
            extra_html=info_box_html,
        )

        return self.build_template(
            title="Thank You for Your Review — BaliRoam",
            header_html=header_html,
            content_html=content_html,
            footer_html=footer_html,
        )
