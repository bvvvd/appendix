package org.bvd.appendix

import java.sql.Connection
import javax.sql.DataSource


class TransactionRunner(private val dataSource: DataSource) {

    fun <T> runInTransaction(block: (connection: Connection) -> T): T {
        dataSource.connection.use { connection ->
            connection.autoCommit = false;
            connection.transactionIsolation = Connection.TRANSACTION_READ_COMMITTED

            try {
                val result = block(connection)
                connection.commit()
                return result
            } catch (e: Exception) {
                System.err.println("Transaction failed: $e")
                connection.rollback()
                throw e
            } finally {
                connection.close()
            }
        }
    }
}
