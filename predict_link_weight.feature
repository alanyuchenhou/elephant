Feature: graph link weight prediction
  As a social network business
  I want to have a regressor that can predict how much a user will like a movie or text another user
  so that I can recommend new movies for a user to watch and new friends for a user to connect to

  Scenario Outline: movie rating prediction
    When the regressor reads a <user>, <movie> pair
    Then it should predict the <rating> the <user> will give to the <movie>

    Examples:
      | user     | movie                    | rating |
      | James    | The Shawshank Redemption | 1      |
      | Mary     | The Dark Knight          | 3      |
      | John     | The Godfather            | 4      |
      | John     | The Dark Knight          | 3      |
      | James    | The Dark Knight          | 5      |
      | Mary     | The Godfather            | 4      |
      | Mary     | The Shawshank Redemption | 2      |
      | Patricia | The Godfather            | 5      |

  Scenario Outline: user messaging volume prediction
    When the regressor reads a <user A>, <user B> pair
    Then it should predict the <number of messages per week> <user A> will send to <user B>

    Examples:
      | user A   | user B   | number of messages per week |
      | James    | Mary     | 87                          |
      | James    | Patricia | 346                         |
      | John     | Patricia | 563                         |
      | James    | Mary     | 42                          |
      | Patricia | John     | 12                          |
      | James    | Mary     | 49                          |
      | James    | John     | 1                           |
      | Patricia | Mary     | 360                         |
