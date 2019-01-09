# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import datetime

## THINGS TO CONFIGURE
## ---------------------------------------------------------------------

AUTHOR = u'A blog-o-matic user'
SITENAME = u'My personal blog with blog-o-matic'
SITESUBTITLE = u'This is where I do test things'
SITEURL = u'/'
DEFAULT_LANG = u'en'
DEFAULT_CATEGORY = 'blog'


LANDING_PAGE_ABOUT = {'title': 'My blog',
                      'details': """<p>This website contains Info that might be interesting for you, enjoy!</p>"""}


PROJECTS = [{'name': 'Blog-o-matic', 'url': 'https://github.com/iranzo/blog-o-matic',
             'description': 'Canned blog automation for quickly setting up a blog with Pelican'},
            {'name': 'Citellus', 'url': 'https://citellus.org',
             'description': 'Troubleshooting automation tool with easy to contribute rules'},
            {'name': 'Pablo Iranzo Blog', 'url': 'https://iranzo.github.io',
             'description': 'Other projects at Github website'},
            {'name': 'Redken Telegram BOT', 'url': 'https://t.me/redken_bot',
             'description': 'A Telegram bot with support for Karma, RSS Feeds, Quotes, etc'}]


# LINKS = (('Redken on telegram', 'https://t.me/redken_bot'),
#          ('RHJobs channel on TG', "https://t.me/rhjobs"),)

# SOCIAL = (('Iranzo twitter', 'http://twitter.com/iranzop'),
#           ('Iranzo github', 'http://github.com/iranzo'),)


# TWITTER_USERNAME = "fillit"
# Update if you use amazon links
AMAZON_ONELINK = "b63a2115-85f7-43a9-b169-5f4c8c275655"

# GOOGLE_ANALYTICS tracking ID
# GOOGLE_ANALYTICS = "UA-81705-12"

## Configure if you use Disqus for comments
# DISQUS_SITENAME = "iranzo-github-io"
# DISQUS_DISPLAY_COUNTS = True


# Customize the files for Claiming your website on search engines
EXTRA_PATH_METADATA = {
    # Customize filename once done the git mv step as instructed on README
    'extra/google3bc953001343abe6': {'path': 'google3bc953001343abe6.html'},
    'extra/BingSiteAuth.xml': {'path': 'BingSiteAuth.xml'}
}

STATIC_PATHS = [
    'images',
    'extra/google3bc953001343abe6', ## Customize filename once done the git mv step as instructed on README
    'extra/BingSiteAuth.xml'
]


## ONLY TOUCH IF YOU KNOW WHAT YOU'RE DOING!
## ---------------------------------------------------------------------

PATH = 'content'

TIMEZONE = 'Europe/Madrid'

# Put as draft content in the future
WITH_FUTURE_DATES = False

# Put full text in RSS feed
RSS_FEED_SUMMARY_ONLY = False

# Feed generation is usually not desired when developing

FEED_ALL_ATOM = 'feeds/all.atom.xml'
FEED_ALL_RSS = 'feeds/all.rss'

CATEGORY_FEED_ATOM = 'feeds/{slug}.atom.xml'
CATEGORY_FEED_RSS = 'feeds/{slug}.rss'
TRANSLATION_FEED_ATOM = 'feeds/{lang}.atom.xml'
TRANSLATION_FEED_RSS = 'feeds/{lang}.rss'
AUTHOR_FEED_ATOM = 'feeds/{slug}.atom.xml'
AUTHOR_FEED_RSS = 'feeds/{slug}.rss'
TAG_FEED_ATOM = 'feeds/tag_{slug}.atom.xml'
TAG_FEED_RSS = 'feeds/tag_{slug}.rss'

DISPLAY_PAGES_ON_MENU = True

CACHE_CONTENT = False
CACHE_PATH = '.cache'
LOAD_CONTENT_CACHE = False

# Plugins
PLUGIN_PATHS = ['plugins']

PLUGINS = ['sitemap', 'extract_toc', 'tipue_search', 'liquid_tags.img',
           'neighbors', 'render_math', 'related_posts', 'share_post',
           'series', 'assets', 'post_stats']

           # 'better_codeblock_line_numbering'
           # 'better_figures_and_images'

THEME = 'themes/elegant'

#elegant
TYPOGRIFY = True
RECENT_ARTICLE_SUMMARY = True
RESPONSIVE_IMAGES = True

MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {
            'css_class': 'highlight',
            'linenums': True
        },
        'markdown.extensions.extra': {},
        'markdown.extensions.toc': {
            'permalink': 'true'
        },
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

DIRECT_TEMPLATES = (('index', 'tags', 'categories','archives', 'search', '404'))

# Elegant Labels
SOCIAL_PROFILE_LABEL = u'Stay in Touch'
RELATED_POSTS_LABEL = 'Keep Reading'
SHARE_POST_INTRO = 'Like this post? Share on:'
COMMENTS_INTRO = u''

FILENAME_METADATA = '(?P<date>\d{4}-\d{2}-\d{2})-(?P<slug>.*)'
USE_FOLDER_AS_CATEGORY = False

SEARCH_BOX = False

# URL Settings to be compatible with octopress
ARTICLE_URL = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/'
ARTICLE_SAVE_AS = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}/index.html'

ARTICLE_LANG_URL = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}-{lang}/'
ARTICLE_LANG_SAVE_AS = 'blog/{date:%Y}/{date:%m}/{date:%d}/{slug}-{lang}/index.html'

YEAR_ARCHIVE_URL = 'blog/archive/{date:%Y}/'
YEAR_ARCHIVE_SAVE_AS = 'blog/archive/{date:%Y}/index.html'

MONTH_ARCHIVE_URL = 'blog/archive/{date:%Y}/{date:%m}/'
MONTH_ARCHIVE_SAVE_AS = 'blog/archive/{date:%Y}/{date:%m}/index.html'

CATEGORY_URL = 'blog/category/{slug}/'
CATEGORY_SAVE_AS = 'blog/category/{slug}/index.html'

TAG_URL = 'blog/tag/{slug}/'
TAG_SAVE_AS = 'blog/tag/{slug}/index.html'

PAGE_URL = '{slug}/'
PAGE_SAVE_AS = '{slug}/index.html'

AUTHOR_SAVE_AS = ''
AUTHORS_SAVE_AS = ''

ARCHIVES_URL = 'blog/archives/'
ARCHIVES_SAVE_AS = 'blog/archives/index.html'

CATEGORIES_URL = 'blog/categories/'
CATEGORIES_SAVE_AS = 'blog/categories/index.html'

TAGS_URL = 'blog/tags/'
TAGS_SAVE_AS = 'blog/tags/index.html'

TAGS_URL = 'tags'
TAGS_SAVE_AS = 'tags/index.html'
AUTHORS_URL = 'authors'
AUTHORS_SAVE_AS = 'authors/index.html'
CATEGORIES_URL = 'categories'
CATEGORIES_SAVE_AS = 'categories/index.html'
ARCHIVES_URL = 'archives'
ARCHIVES_SAVE_AS = 'archives/index.html'

DEFAULT_PAGINATION = 5
DEFAULT_ORPHANS = 0

PAGINATION_PATTERNS = (
    (1, '{base_name}/', '{base_name}/index.html'),
    (2, '{base_name}/page/{number}/', '{base_name}/page/{number}/index.html'),
)


# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

# better codeblock
MARKDOWN = {
    'extension_configs': {
        'markdown.extensions.codehilite': {'css_class': 'highlight', 'linenums': False},
        'markdown.extensions.extra': {},
        'markdown.extensions.meta': {},
    },
    'output_format': 'html5',
}

# sitemap
SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.5,
        'indexes': 0.5,
        'pages': 0.5
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly'
    }
}

SITE_UPDATED = datetime.date.today()

# use those if you want pelican standard pages to appear in your menu
MENU_INTERNAL_PAGES = (
    ('Tags', TAGS_URL, TAGS_SAVE_AS),
    ('Archives', ARCHIVES_URL, ARCHIVES_SAVE_AS),
)