import unittest

import generatefeedvector


class FeedGeneratorTest(unittest.TestCase):
    def test_get_words(self):
    	'''
        Test the count of words in a string, exluding the html
        tags
        '''
        d = '<hallo>this</hallo> is text'
        expected_words = ['this', 'is', 'text']
        self.assertEquals(expected_words, generatefeedvector.getwords(d))

    def test_list_of_words(self):
        '''
        Tests the words that meet the required appearance in the blogs
        to be used for analysis. The appearance should be between 10%
        and 50% of all the blogs
        '''
        apcount = {
            'this': 5,
            'extremely': 1,
            'nice': 1,
            'okay': 1,
            'is': 5,
            'bad': 1,
            'the': 10,
            'good': 2,
            'a': 9}

        words_list = generatefeedvector.list_of_words(apcount, 11)
        self.assertEquals(3, len(words_list))
        self.assertListEqual(sorted(['this', 'is', 'good']), sorted(words_list))

if __name__ == '__main__':
    unittest.main()
