import feedparser
import collections
import re


def getwords(html):
    '''
    A function that returns the words in string, excluding the html tags
    html - the string which may contain html tags
    '''
    text = re.compile(r'<[^>]+>').sub('', html)  # stripping off html tags
    words = re.compile(r'[^A-z^a-z]+').split(text)  # spilitting the string using non-alphabet characters
    return [word.lower() for word in words if word]  # retruning the list of the words in lower case


def getwordcounts(url):
    '''
    Get the counts of words in a blog
    url - the url of the blog
    returns a tuple of the title of the blog and the dict of the words as keys
    and their counts as values
    '''
    d = feedparser.parse(url)  # passing an RSS parser
    wc = collections.defaultdict(int)

    for e in d.entries:
        if 'summary' in e:
            summary = e.summary
        else:
            summary = e.description

        words = getwords('{} {}'.format(e.title, summary))
        for word in words:
            # encountering the key of the dict for the first time calls int() 
            # to supply a default value of zero which will the then be 
            # incremented. subsequent access to the key will function as a
            # a normal dict
            wc[word] += 1

    if 'title' not in d.feed:
        print 'Invalid url', url
        return 'bogus data', wc
    return d.feed.title, wc


def main():
    apcount = collections.defaultdict(int)
    wordcounts = {}
    feedlist = open('feedlist.txt').readlines()
    for url in feedlist:
        title, wc = getwordcounts(url)
        wordcounts[title] = wc
        for word, count in wc.iteritems():
            if count > 1:
                apcount[word] += 1

    wordlist = []
    for w, bc in apcount.iteritems():
        frac = float(bc)/len(feedlist)
        if 0.1 < frac < 0.5:
            wordlist.append(w)

    out = file('blogdata.txt', 'w')
    out.write('Blog')
    for w in wordlist:
        out.write('\t' + w)
    out.write('\n')
    for blogname, counts in wordcounts.iteritems():
        out.write(blogname)
        for w in wordlist:
            if w in counts:
                out.write('\t%d' % counts[w])
            else:
                out.write('\t0')
        out.write('\n')

if __name__ == '__main__':
    main()
