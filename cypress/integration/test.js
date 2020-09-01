describe('E2E Tests', () => {
  it('visits Playground without errors', () => {
    cy.visit('localhost:5000');
  });

  describe('UI interaction', () => {
    it('should control the timeline', () => {
      cy.get('#play-pause-button').click().click();

      cy.get('#next-step-button').click();

      cy.get('#reset-button').click();
    });

    it('should change the parameters', () => {
      cy.get('select#learningRate')
        .select('0.01', { force: true })
        .invoke('val')
        .should('eq', '0.01');

      cy.get('select#activations')
        .select('relu', { force: true })
        .invoke('val')
        .should('eq', 'relu');

      cy.get('select#regularizations')
        .select('L1', { force: true })
        .invoke('val')
        .should('eq', 'L1');

      cy.get('select#regularRate')
        .select('0.1', { force: true })
        .invoke('val')
        .should('eq', '0.1');

      cy.get('select#problem')
        .select('regression', { force: true })
        .invoke('val')
        .should('eq', 'regression');

      cy.get('select#problem').select('classification', { force: true });
    });

    it('should change the data type and generate a new data set', () => {
      cy.get('canvas[data-dataset="gauss"]').click();

      cy.get('input#percTrainData')
        .invoke('val', 90)
        .trigger('change')
        .invoke('val', 0)
        .trigger('change');

      cy.get('input#noise')
        .invoke('val', 50)
        .trigger('change')
        .invoke('val', 0)
        .trigger('change');

      cy.get('input#batchSize')
        .invoke('val', 30)
        .trigger('change')
        .invoke('val', 1)
        .trigger('change');

      cy.get('#data-regen-button').click();
    });

    it('should reconfigure the neural network', () => {
      // Double-click all nodes (features and neurons) in the the network
      cy.get('#network canvas').each((canvas) => {
        canvas.click().click();
      });

      cy.get('.ui-numHiddenLayers button').each((button) => {
        button.click();
      });

      cy.get('.plus-minus-neurons button').each((button) => {
        button.click();
      });
    });

    it('should display the output in a different way', () => {
      cy.get('.column.output input[type="checkbox"]').each((checkbox) => {
        checkbox.click().click();
      });
    });
  });
});
