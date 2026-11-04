"""Base HTML document template for emails."""


BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>{title}</title>
    <style>
        /* Reset styles */
        body, table, td, p, a, li, blockquote {{
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
        }}
        table, td {{
            mso-table-lspace: 0pt;
            mso-table-rspace: 0pt;
        }}
        img {{
            -ms-interpolation-mode: bicubic;
            border: 0;
            height: auto;
            line-height: 100%;
            outline: none;
            text-decoration: none;
        }}
        /* Mobile-first base styles */
        body {{
            font-family: {font_stack};
            background-color: {color_bg_page};
            margin: 0;
            padding: 0;
            color: {color_text_primary};
        }}
        .container {{
            width: 100%;
            max-width: {max_width};
            margin: {margin_mobile};
            background-color: {color_bg_container};
            border-radius: {border_radius};
            overflow: hidden;
            box-shadow: {box_shadow};
            border: 1px solid {color_border_container};
        }}
        .header {{
            background: {header_background};
            padding: {padding_header_mobile};
            text-align: center;
        }}
        .header img {{
            max-width: {logo_width};
            height: auto;
        }}
        .content {{
            padding: {padding_mobile};
            text-align: center;
        }}
        .content h1 {{
            color: {color_primary};
            font-size: {font_h1_mobile};
            margin: 16px 0 20px 0;
            line-height: 1.3;
        }}
        .content p {{
            line-height: {line_height};
            color: {color_text_secondary};
            font-size: {font_base_mobile};
            margin: 0 0 {para_margin} 0;
        }}
        .button-container {{
            margin-top: {button_margin};
        }}
        .button {{
            color: #ffffff !important;
            text-decoration: none;
            font-weight: bold;
            font-size: {font_base_mobile};
            padding: {button_padding};
            border-radius: {button_radius};
            background-color: {color_accent};
            display: inline-block;
            min-height: {button_min_height};
            box-sizing: border-box;
            transition: all 0.3s;
            box-shadow: -1px -1px 1px rgba(255, 255, 255, 0.46), 1px 1px 1px rgba(77, 77, 81, 0.8);
            text-shadow: 2px 1px 4px rgba(34, 33, 33, 0.5);
        }}
        .button:hover {{
            background-color: #ff6b33;
            box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.5), inset -2px -2px 5px rgba(255, 253, 225, 0.749);
        }}
        .footer {{
            background-color: {color_bg_footer};
            padding: {padding_footer};
            text-align: center;
            font-size: {font_footer};
            color: {color_text_footer};
            border-top: 1px solid {color_border};
        }}
        .footer p {{
            margin: 5px 0;
            color: {color_text_footer};
        }}
        .security-box {{
            margin: 24px 0 0 0;
            padding: {security_padding};
            background-color: {color_warning_bg};
            border-radius: {security_radius};
            border: {security_border};
            text-align: left;
        }}
        .security-box p {{
            margin: 0;
            font-size: {font_small};
            color: {color_warning_text};
        }}
        .info-box {{
            margin: 24px 0;
            padding: {info_padding};
            background-color: {color_bg_footer};
            border-radius: {info_radius};
            border-left: {info_border_left};
            text-align: left;
        }}
        .info-box p {{
            margin: 0;
            font-size: {font_small};
            color: {color_text_secondary};
        }}
        .info-box h3 {{
            margin: 0 0 16px 0;
            color: {color_primary};
            font-size: 18px;
        }}
        .content p.expiry-note {{
            margin-top: 30px;
            font-size: {font_small};
            color: {color_text_muted};
        }}
        /* Desktop enhancements */
        @media screen and (min-width: 600px) {{
            .container {{
                margin: {margin_desktop};
            }}
            .header {{
                padding: {padding_header_desktop};
            }}
            .content {{
                padding: {padding_desktop};
            }}
            .content h1 {{
                font-size: {font_h1_desktop};
                margin-bottom: 20px;
            }}
            .content p {{
                font-size: {font_base_desktop};
                margin-bottom: 16px;
            }}
            .button {{
                font-size: {font_base_desktop};
            }}
        }}
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {{
            body {{
                background-color: #1a1a1a;
            }}
            .container {{
                background-color: #2d2d2d;
            }}
            .content h1 {{
                color: #4a90e2;
            }}
            .content p {{
                color: #cccccc;
            }}
            .security-box p {{
                color: {color_warning_text};
            }}
            .info-box p {{
                color: {color_text_secondary};
            }}
            .content p.expiry-note {{
                color: #cccccc;
            }}
            .footer {{
                background-color: #252525;
                border-top-color: #3d3d3d;
            }}
        }}
    </style>
</head>
<body>
    {body_content}
</body>
</html>
"""
