import hu_core_news_lg


def main():
    nlp = hu_core_news_lg.load()
    test = "Példa mondat."
    doc = nlp(test)


if __name__ == '__main__':
    main()